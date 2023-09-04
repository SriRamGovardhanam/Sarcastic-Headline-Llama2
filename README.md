## Fine tuning Llama 2 to respond in funny, sarcastic way using a generated dataset 
- The main idea behind the model is to add behaviour to an LLM so that for a given input(news headline) the model responds back with output(sarcastic_headline) in a funny, sarcastic way.<br>
- All the existing open datasets available related to sarcasm are either extracted from social media like twitter or reddit which were mostly replies to parent post or just a labelled dataset which have sarcastic, non-sarcastic sentences. <br>we are looking for dataset which has normal sentence and corresponding sarcastic version for the model to understand. 
- We can generate such dataset using a LLM by giving a random sentence and ask the model to generate sarcastic version of it. Once we get the generated dataset, we can fine tune a LLM model and to give sarcastic response.
### Model Details
We are using Llama 2 13B version to generate the sarcastic sentence by using an appropriate prompt template, for the input sentences we are referring to a news headline category dataset. once we generate dataset, we format the dataset and do PEFT on pretrained Llama 2 7B weights. <br>

The fine tuned model can behave sarcastically and generate satirical responses. To ensure the quality and diversity of the training data, we are picking news headline category dataset so that we can cover multiple different random sentences without worrying about grammatic mistakes in input sentence.

### Model weights and checkpoints
- Refer to this HF repo: https://huggingface.co/Sriram-Gov/Sarcastic-Headline-Llama2 <br>
- Llama2 Generated dataset can be found in this repo. The code to generate sarcastic_response for a given headline can be found in ***Sarcasm_ETL_&_data_generation.ipynb*** notebook file <br>
Fine tuning the above generated data and Inferencing on output weights can be found in ***LLM_sarcasm_headlines_fine_tuning.ipynb*** file, additionally we can upload these weights as model card into Huggingface repo.<br>
- Start using the uploaded checkpoints for a quick test on this finetuned model, Inference code can be found in ***Infer_with_adapters.ipynb*** notebook file (use PEFT adapters directly for playing with the finetuned Llama2 as present in the notebook file) <br><br>
Note:
  - If you want to push it in HF (huggingface), then its better to do while fine tuning by using parameters
  ```
  --push_to_hub
  --repo_id your_repo_id
  ```
  - If you dont want to push it in HF but want to use it as HF plug n play type model in local then you can specify below param while training ```--merge-adapters```
  - Since merging base model with adapter is a pretty cpu intensie task, it can definelty crash the existing session if you are using colab. It almost used 35GB CPU RAM when i merged it seperately.
  Colab pro version will be needed to have that much of a RAM.

### Uses
- ***Enhanced Natural Language Understanding:*** In applications like chatbots or virtual assistants, a model trained to understand sarcasm can provide more contextually relevant responses, improving user interactions.
- ***Niche applications:*** For some websites like TheOnion, we may able to support/improve writers ability. Social media platforms to engage users with witty and sarcastic responses.

### Training Details
- use LLM_sarcasm_headlines_fine_tuning notebook file to load the processed data and start fine tuning.
```
autotrain llm --train --project_name 'sarcastic-headline-gen' --model TinyPixel/Llama-2-7B-bf16-sharded \
--data_path '/content/sarcastic-headline' \
--use_peft \
--use_int4 \
--learning_rate 2e-4 \
--train_batch_size 8 \
--num_train_epochs 8 \
--trainer sft \
--model_max_length 340 > training.log &
```

### Results
Input headline:  **i love to eat pizza**
<br>Input Formatted Template to the fine tuned LLM:
```
You are a savage, disrespectful and witty agent. You convert below news headline into a funny, humiliating, creatively sarcastic news headline while still maintaining the original context.
### headline: i love to eat pizza
### sarcastic_headline:
```
<br>Output after Inferencing:
```
You are a savage, disrespectful and witty agent. You convert below news headline into a funny, humiliating, creatively sarcastic news headline while still maintaining the original context.
### headline: i love to eat pizza
### sarcastic_headline:  I love to eat pizza too! But only if it's delivered by a hot waitress wearing a tight dress and high heels.
```

#### Summary
The primary purpose of this model is often to generate humor and entertainment. It can be used in chatbots, virtual assistants, or social media platforms to engage users with witty and sarcastic responses.

### Model Objective
This model is not intended to target specific race, gender, region etc., Sole purpose of this model is to understand LLM's and tap the LLM's ability to entertain, engage.

### Compute Infrastructure
Google colab pro is needed if you are planning to tune more than 5 epochs for a 2100 samples of model_max_length < 650.

## Citation
Configuring Llama2 in google colab was referred from https://github.com/R3gm/InsightSolver-Colab/blob/main/LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat.ipynb <br>
One line Fine tuning code was referred from HF autotrain-advanced colab examples https://github.com/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain_LLM.ipynb <br>
The source dataset - news headlines are taken from https://www.kaggle.com/datasets/rmisra/news-category-dataset <br>

### Contact
Sriram Govardhanam
http://www.linkedin.com/in/SriRamGovardhanam
