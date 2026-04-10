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
        self.successful_guideline = None
        self.failed_guideline = None

    def update_entities(self, entities):
        self.entities = entities

    def extract_retained_components(self):
        """Split retained entities into first and last name pools."""
        first_names = set()
        last_names = set()
        for e in self.entities:
            parts = e.strip().split()
            if len(parts) >= 2:
                first_names.add(parts[0])
                last_names.add(parts[-1])
        return sorted(first_names), sorted(last_names)

    def generate_first_names(self, n=50, feedback=None):
        """Generate candidate first names using LLM + retained seeds."""
        retained_firsts, _ = self.extract_retained_components()

        positive_guide = ""
        negative_guide = ""
        if feedback:
            if feedback.get("good"):
                positive_guide = "\nFirst names that scored well (generate similar ones):\n" + \
                    "\n".join(f"- {x}" for x in feedback["good"][:15])
            if feedback.get("bad"):
                negative_guide = "\nFirst names that scored poorly (avoid these patterns):\n" + \
                    "\n".join(f"- {x}" for x in feedback["bad"][:15])

        sample_firsts = "\n".join(f"- {x}" for x in retained_firsts[:25])
        prompt = f"""Task: Generate {n} plausible fictional FIRST NAMES (given names only, no surnames).

            These are example first names from the dataset for style reference:
            {sample_firsts}

            Rules:
            - Only output first names (single words), one per line
            - Format: - Firstname
            - Include diverse cultural origins (Hispanic, African, Asian, European, Middle Eastern, etc.)
            - No numbering, no labels, no explanations
            - Do not repeat the examples above
            {positive_guide}
            {negative_guide}

            Output:"""

        all_names = set(retained_firsts)

        # Multiple LLM calls with higher temperature for diversity
        for _ in range(3):
            raw = self._query_local_LLM_diverse(prompt.strip())
            parsed = self._parse_single_names(raw)
            all_names.update(parsed)

        return sorted(all_names)

    def generate_last_names(self, n=50, feedback=None):
        """Generate candidate last names using LLM + retained seeds."""
        _, retained_lasts = self.extract_retained_components()

        positive_guide = ""
        negative_guide = ""
        if feedback:
            if feedback.get("good"):
                positive_guide = "\nLast names that scored well (generate similar ones):\n" + \
                    "\n".join(f"- {x}" for x in feedback["good"][:15])
            if feedback.get("bad"):
                negative_guide = "\nLast names that scored poorly (avoid these patterns):\n" + \
                    "\n".join(f"- {x}" for x in feedback["bad"][:15])

        sample_lasts = "\n".join(f"- {x}" for x in retained_lasts[:25])
        prompt = f"""Task: Generate {n} plausible fictional LAST NAMES (surnames only, no first names).

            These are example last names from the dataset for style reference:
            {sample_lasts}

            Rules:
            - Only output last names (single words or hyphenated), one per line
            - Format: - Lastname
            - Include diverse cultural origins (Hispanic, African, Asian, European, Middle Eastern, etc.)
            - No numbering, no labels, no explanations
            - Do not repeat the examples above
            {positive_guide}
            {negative_guide}

            Output:"""

        all_names = set(retained_lasts)

        for _ in range(3):
            raw = self._query_local_LLM_diverse(prompt.strip())
            parsed = self._parse_single_names(raw)
            all_names.update(parsed)

        return sorted(all_names)

    def _query_local_LLM_diverse(self, prompt):
        """Query local LLM with higher temperature for diversity."""
        outputs = self.pipe(
            prompt, max_new_tokens=500, do_sample=True,
            temperature=0.7, top_p=0.95, repetition_penalty=1.2,
        )
        return outputs[0]["generated_text"][len(prompt):].strip()

    def _parse_single_names(self, raw_text):
        """Parse single-word names from LLM output."""
        names = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Stop if LLM starts generating code or explanations
            if line.startswith("```") or line.startswith("Here ") or line.startswith("import "):
                break
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if not line:
                continue
            # Accept single names or hyphenated names
            if line[0].isupper() and len(line) >= 2 and not any(ch.isdigit() for ch in line):
                # Take first word only (in case LLM adds extra)
                word = line.split()[0].strip(".,;:")
                if len(word) >= 2 and word[0].isupper():
                    names.append(word)
        return list(dict.fromkeys(names))

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
    
    def build_generation_prompt(self, style_summary = str, n_to_generate=20,
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

        # print("=======(Cancelled) Building prompt that extracts style of retained entities...========")
        # style_prompt = self.build_style_prompts()
        # print("Done! Style Prompt:", style_prompt)

        # print("=======(Cancelled) Querying LLM...========")
        # style_summary = self.query_local_LLM(style_prompt)
        # print("Done! Style Summary is:", style_summary)

        print("=======Building prompt for entity generation...========")
        generation_prompt = self.build_generation_prompt() # add style summary if exists
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

        

