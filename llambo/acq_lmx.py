import os
import random
import math
import time
import openai
import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientSession


class LLM_ACQ_LMX:
    def __init__(self, task_context, n_candidates, n_templates, lower_is_better, 
                 jitter=False, rate_limiter=None, warping_transformer=None, chat_engine=None, 
                 prompt_setting=None, shuffle_features=False):
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = int(n_candidates/n_templates)
        self.lower_is_better = lower_is_better
        self.apply_jitter = jitter
        if rate_limiter is None:
            self.rate_limiter = None
        else:
            self.rate_limiter = rate_limiter
        self.warping_transformer = warping_transformer
        self.apply_warping = warping_transformer is not None
        self.chat_engine = chat_engine
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

    def _jitter(self, desired_fval):
        if not self.apply_jitter:
            return desired_fval

        assert hasattr(self, 'observed_best'), 'observed_best must be set before calling _jitter'
        assert hasattr(self, 'observed_worst'), 'observed_worst must be set before calling _jitter'
        assert hasattr(self, 'alpha'), 'alpha must be set before calling _jitter'

        jittered = np.random.uniform(
            low=min(desired_fval, self.observed_best), 
            high=max(desired_fval, self.observed_best), 
            size=1
        ).item()

        return jittered

    def create_crossover_prompt(self, examples):
        """Creates a crossover prompt using the provided examples."""
        prompt = ""
        for candidate in examples:
            prompt += candidate + "\n"
        return prompt

    async def generate_crossover_prompt(self, observed_configs, examples=5, temp=1.0, batch_size=2, max_tokens=1200):
        """Generates new configurations using GPT-3.5 Turbo via OpenAI API."""
        chosen_examples = observed_configs.sample(n=examples, replace=False).to_dict(orient='records')
        prompt = self.create_crossover_prompt([str(example) for example in chosen_examples])
        formatted_prompt = (f"Here are some examples of hyperparameter settings for decision trees in the format "
                        "`key: value`:\n{prompt}\n\n"
                        "Please generate 20 new candidate configurations for decision trees. "
                        "Each configuration should follow this format:\n"
                        "max_depth: value, max_features: value, min_impurity_decrease: value, "
                        "min_samples_leaf: value, min_samples_split: value, min_weight_fraction_leaf: value.\n"
                        "Please provide the configurations in the same format as shown above and it should contain all the mentioned hyperparams.")

        responses = []
        for _ in range(batch_size):
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": formatted_prompt}],
                max_tokens=max_tokens,
                temperature=temp,
            )
            responses.append(response['choices'][0]['message']['content'].strip())

        return responses

    def _count_decimal_places(self, n):
        s = format(n, '.10f')
        if '.' not in s:
            return 0
        n_dp = len(s.split('.')[1].rstrip('0'))
        return n_dp

    def _prepare_configurations_acquisition(self, observed_configs=None, observed_fvals=None, seed=None,
                                            use_feature_semantics=True, shuffle_features=False):
        examples = []
        
        if seed is not None:
            np.random.seed(seed)
            shuffled_indices = np.random.permutation(observed_configs.index)
            observed_configs = observed_configs.loc[shuffled_indices]
            if observed_fvals is not None:
                observed_fvals = observed_fvals.loc[shuffled_indices]
        else:
            if type(observed_fvals) == pd.DataFrame:
                observed_fvals = observed_fvals.sort_values(by=observed_fvals.columns[0], ascending=not self.lower_is_better)
                observed_configs = observed_configs.loc[observed_fvals.index]

        if shuffle_features:
            np.random.seed(0)
            shuffled_columns = np.random.permutation(observed_configs.columns)
            observed_configs = observed_configs[shuffled_columns]

        if observed_configs is not None:
            hyperparameter_names = observed_configs.columns
            for index, row in observed_configs.iterrows():
                row_string = '## '
                for i in range(len(row)):
                    hyp_type = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][0]
                    hyp_transform = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][1]

                    if use_feature_semantics:
                        row_string += f'{hyperparameter_names[i]}: '
                    else:
                        row_string += f'X{i+1}: '

                    lower_bound = self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][0] \
                        if hyp_type in ['int', 'float'] else \
                        self.task_context['hyperparameter_constraints'][hyperparameter_names[i]][2][1]
                    
                    n_dp = self._count_decimal_places(lower_bound)
                    value = row[i]
                    if self.apply_warping:
                        if hyp_type == 'int' and hyp_transform != 'log':
                            row_string += str(int(value))
                        elif hyp_type == 'float' or hyp_transform == 'log':
                            row_string += f'{value:.{n_dp}f}'
                        elif hyp_type == 'ordinal':
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value
                    else:
                        if hyp_type == 'int':
                            row_string += str(int(value))
                        elif hyp_type in ['float', 'ordinal']:
                            row_string += f'{value:.{n_dp}f}'
                        else:
                            row_string += value

                    if i != len(row)-1:
                        row_string += ', '
                row_string += ' ##'
                example = {'Q': row_string}
                if observed_fvals is not None:
                    row_index = observed_fvals.index.get_loc(index)
                    perf = f'{observed_fvals.values[row_index][0]:.6f}'
                    example['A'] = perf
                examples.append(example)
        elif observed_fvals is not None:
            examples = [{'A': f'{observed_fvals:.6f}'}]
        else:
            raise Exception("Both observed_configs and observed_fvals cannot be None")
            
        return examples

    def _convert_to_json(self, response_str):
        '''Parse LLM response string into JSON.'''
        print("Response to convert:", response_str)
        
        
        response_lines = response_str.strip().split('\n')
        for line in response_lines:
            response_json = {}
        
        # Split each line by commas to get key-value pairs
        pairs = line.split(',')
        for pair in pairs:
            try:
                # Split each pair by colon and strip whitespace
                key, value = [x.strip() for x in pair.split(':', 1)]
                
                # Convert value to float
                response_json[key] = float(value)
            except ValueError as e:
                print(f"Error parsing pair '{pair}': {e}")
            except IndexError as e:
                print(f"Error splitting pair '{pair}': {e}")
        
        # Append the dictionary to the result list if it's not empty
    
    # Return the list of dictionaries as JSON
        return response_json
        #pairs = response_str.split(',')
        #response_json = {}
        #for pair in pairs:
        #    key, value = [x.strip() for x in pair.split(':')]
        #    response_json[key] = float(value)
        #return response_json

    def _filter_candidate_points(self, observed_points, candidate_points, precision=8):
        '''Filter candidate points that already exist in observed points. Also remove duplicates.'''
        rounded_observed = [{key: round(value, precision) for key, value in d.items()} for d in observed_points]
        rounded_candidate = [{key: round(value, precision) for key, value in d.items()} for d in candidate_points]
        filtered_candidates = [x for i, x in enumerate(candidate_points) if rounded_candidate[i] not in rounded_observed]

        def is_within_range(value, allowed_range):
            value_type, transform, search_range = allowed_range
            if value_type == 'int':
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                    return min_val <= value <= max_val
                else:
                    return min_val <= value <= max_val and int(value) == value
            elif value_type == 'float':
                [min_val, max_val] = search_range
                if transform == 'log' and self.apply_warping:
                    min_val = np.log10(min_val)
                    max_val = np.log10(max_val)
                return min_val <= value <= max_val
            elif value_type == 'ordinal':
                return any(math.isclose(value, x, abs_tol=1e-2) for x in search_range)
            else:
                raise Exception('Unknown hyperparameter value type')

        def is_dict_within_ranges(d, ranges_dict):
            return all(key in ranges_dict and is_within_range(value, ranges_dict[key]) for key, value in d.items())

        filtered_candidates = [d for d in filtered_candidates if is_dict_within_ranges(d, self.task_context['hyperparameter_constraints'])]

        filtered_candidates = [dict(t) for t in {tuple(d.items()) for d in filtered_candidates}]
        return filtered_candidates

    async def acquire(self, observed_configs, observed_fvals, alpha=0.8, alpha_range=(-6, 6)):
        """Asynchronous function to acquire new candidate configurations."""
        start_time = time.time()

        self.alpha = alpha

        # Scale fvals and get log-transformed values
        log_scaled_fvals = np.log10(1e-12 + observed_fvals.values + np.abs(np.min(observed_fvals.values)))
        N = len(log_scaled_fvals)
        if self.lower_is_better:
            z_score = (log_scaled_fvals - np.mean(log_scaled_fvals)) / np.std(log_scaled_fvals)
        else:
            z_score = -(log_scaled_fvals - np.mean(log_scaled_fvals)) / np.std(log_scaled_fvals)
        scaled_target = np.mean(log_scaled_fvals) + self.alpha * np.std(log_scaled_fvals) * np.sqrt(N)
        jittered_target = self._jitter(scaled_target)
        self.observed_best = np.min(log_scaled_fvals)
        self.observed_worst = np.max(log_scaled_fvals)

        if jittered_target < self.observed_best:
            jittered_target = self.observed_best

        self.target_fval = 10 ** jittered_target
        self.target_fval = np.clip(self.target_fval, alpha_range[0] * N, alpha_range[1] * N)

        # Generate crossover points based on observed configurations
        crossover_points = await self.generate_crossover_prompt(observed_configs, examples=5, temp=1.0, batch_size=self.n_gens)

        # Convert string-based candidate points to dictionaries
        candidate_points = []
        for response in crossover_points:
            candidate_dict = self._convert_to_json(response)
            candidate_points.append(candidate_dict)

        # Filter and remove duplicate candidates
        final_candidates = self._filter_candidate_points(observed_configs.to_dict(orient='records'), candidate_points)

        print(f'Acquisition completed in {time.time() - start_time:.2f} seconds')

        return final_candidates