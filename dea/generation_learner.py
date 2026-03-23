import re
import unicodedata
from openai import OpenAI
from difflib import SequenceMatcher
import torch
from transformers import pipeline

class GenerationLearner:
    def __init__(self, retained_entities):
        self.entities = retained_entities
        # self.client = OpenAI()
        self.client = None
        self.pipe = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def build_style_prompts(self) -> str:
        '''
        Goal: Formulate prompts that extract and describe the styles of the retained entities
        '''
        ent_string = "\n".join(f"- {x}" for x in self.entities)

        prompt = f"""
            You are analyzing a synthetic dataset of fictional entity names.

            Below are retained entities from the dataset:
            {ent_string}

            Infer the latent naming distribution from these examples.

            Describe briefly:
            1. common formatting patterns
            2. diversity of cultural or linguistic styles
            3. punctuation patterns (hyphen, apostrophe, spaces, accents)
            4. realism level (natural-looking vs synthetic-looking)
            5. token count / name length tendencies

            Keep the analysis concise and focused on generation style.
        """

        return prompt.strip()
    
    def build_generation_prompt(self, style_summary, n_to_generate=20,
                                successful_guideline: list[str] | None = None,
                                failed_guideline: list[str] | None = None):
        ''' 
            Goal: Build a prompt for the entities using the prompt of the style summary as guidance. 
        '''
        
        positive_guide = ""
        if successful_guideline:
            positive_guide = "\nSuccessful generated probes that caused stronger refusal:\n" + \
                      "\n".join(f"- {x}" for x in successful_guideline[:30])
        negative_guide = ""
        if failed_guideline:
            negative_guide = "\nExamples to avoid repeating or lightly editing:\n" + \
                     "\n".join(f"- {x}" for x in failed_guideline[:30])
        
        prompt1 = f"""
            You are generating fictional entity names for a synthetic QA dataset.

            Retained entities from the dataset:
            {self.entities}

            Style summary:
            {style_summary}

            Generate {n_to_generate} NEW fictional person names that match the same latent naming distribution.

            Requirements:
            - Names must be novel and diverse
            - Do NOT copy retained names
            - Do NOT produce minor edits of retained names
            - Do NOT simply recombine retained first and last names
            - Do NOT use famous real people
            - Stay faithful to the inferred dataset style
            - Output exactly one name per line
            - Output only the names, with no numbering or commentary
            {positive_guide}
            {negative_guide}
        """

        ent_string = "\n".join(f"- {x}" for x in self.entities[:20]) # too many examples break the LLM
        prompt2 = f"""
            Task: Generate {n_to_generate} fictional person names.

            Output format example:
            {ent_string}

            Now generate {n_to_generate} NEW names in the exact same format.

            Rules:
            - Only bullet points.
            - Exactly one name per line.
            - Format: - Firstname Lastname
            - No numbering.
            - No labels.
            - No explanations.
            - No extra text.
            - Do not repeat examples.

            Output:
        """

        return prompt2.strip()
    
    def query_LLM(self, prompt):
        ''' 
            Goal: Query an external LLM with the prompt we created in `build_generation_prompt()`
        '''

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt,
        )

        return response.output_text
    
    def query_local_LLM(self, prompt):
        ''' 
            Goal: Query a local hugging face LLM with the prompt we created in `build_generation_prompt()`

            Pipeline model generates prompt + new_tokens, so we need to extract the new_tokens separately.
        '''

        outputs = self.pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.2, top_p=0.8, repetition_penalty=1.2,)
        return outputs[0]["generated_text"][len(prompt):].strip()
    
    def parse_names(self, raw_text: str) -> list[str]:
        names = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()

            if line:
                names.append(line)

        return list(dict.fromkeys(names))

    def filter_invalids(self, candidates, similarity_threshold: float = 0.88, for_retained: bool = False) -> list[str]:
        '''
            Goal: Filter invalid candidates, just in case the LLM is not listening to the earlier prompt and still output invalid candidates
        '''
        def normalised_name(name: str) -> str:
            name = name.lower().strip()
            name = re.sub(r"\s+", " ", name)
            return name
        
        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, normalised_name(a), normalised_name(b)).ratio()
        
        def is_latin_letter(ch: str) -> bool:
            if not ch.isalpha():
                return False
            return "LATIN" in unicodedata.name(ch, "")
        
        def is_valid_name_format(name: str) -> bool:
            name = name.strip()
            # print(f"===={name}, length: {len(name)}, digits?: {any(ch.isdigit() for ch in name)}, bad_char?: {bad_char}")
            if not name[0].isupper():
                return False
            if len(name) < 6 or len(name) > 60:
                return False
            if any(ch.isdigit() for ch in name):
                return False
            
            bad_char = None
            for ch in name:
                if is_latin_letter(ch):
                    continue
                if ch in {" ", "-", "'", "."}:
                    continue
                bad_char = ch
                break

            if bad_char is not None:
                return False

            return True
        
        filtered = []
        # If to filter retained entities, just check if they are of valid name formats
        if for_retained:
            for cand in candidates:
                if is_valid_name_format(cand):
                    filtered.append(cand)
            return filtered
        
        seen = set()
        retained_norm = [normalised_name(x) for x in self.entities]

        for cand in candidates:
            cand_norm = normalised_name(cand)

            if cand_norm in seen:
                continue
            if not is_valid_name_format(cand):
                continue

            too_close = False
            for ref in retained_norm:
                if similarity(cand_norm, ref) >= similarity_threshold:
                    too_close = True
                    break

            if too_close:
                continue

            seen.add(cand_norm)
            filtered.append(cand)

        return filtered
    
    def overall_run(self):
        ''' The main function to run in this class. '''

        print("=======Filtering Invalid Retained Candidates...========")
        valid_entities = self.filter_invalids(self.entities, for_retained=True)
        self.entities = valid_entities
        # print("Done! Valid retained entities are:", self.entities)

        print("=======Building prompt that extracts style of retained entities...========")
        style_prompt = self.build_style_prompts()
        # print("Done! Style Prompt:", style_prompt)

        print("=======Querying LLM...========")
        style_summary = self.query_local_LLM(style_prompt)
        # print("Done! Style Summary is:", style_summary)

        print("=======Building prompt for entity generation...========")
        generation_prompt = self.build_generation_prompt(style_summary)
        print("Done! Generation Prompt:", generation_prompt)

        print("=======Querying LLM...========")
        response = self.query_local_LLM(generation_prompt)
        print("Done! Response from LLM is:", response)

        print("=======Parsing Names from LLM response...========")
        candidate_entities = self.parse_names(response)
        print("Done! Candidate Entities are:", candidate_entities)

        print("=======Filtering Invalid Candidates...========")
        valid_candidates = self.filter_invalids(candidate_entities)
        print("Done! Valid Candidate Entities are:", valid_candidates)

        return valid_candidates

        

