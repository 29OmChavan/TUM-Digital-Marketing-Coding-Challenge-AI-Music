# Coding Task – AI & Music  

Welcome to the coding task for the **Professorship of Digital Marketing, TUM**.  

The goal is to give you a small but realistic impression of the kind of projects we work on (AI, data pipelines, music/audio) and to let us see how you approach such tasks.  

This is not only for us to test your skills. It’s also an opportunity for you to see if you enjoy this type of work.  

## Task Overview  

You will build a small pipeline with **four steps**:  

0. **Setup**
   - Fork the repository and make it private.

1. **Generate 30 audio tracks**  
   - Use a model of your choice on [Replicate](https://replicate.com/) via their **Python SDK (API)**.  
   - Either write the prompts yourself or use a model from Replicate to generate them.  
   - Save all generated songs as audio files in an **`outputs/audio/`** folder.  
   - Record metadata in a CSV (using Pandas):  
     - `song_id` (e.g. index or uuid)  
     - `prompt`  
     - `model` used  
     - `audio_path`  
     - `generation_time_seconds` (measured for the Replicate call)  

2. **Extract audio features**  
   - For each song, compute **at least 5 audio features** of your choice.  
   - Add the feature values as columns to the same CSV.  

3. **Create 15-second snippets**  
   - Create a 15 seconds audio thumbnail for each song, which could be used to advertise them on social media or preview them on a streaming platform.
   - Default: extract a **random 15s** segment from each track.  
   - You can also propose and implement alternative methods.  
   - Apply a **0.5s fade-in and 0.5s fade-out**.  
   - Save snippets to an **`outputs/snippets/`** folder.  
   - Add `snippet_path`, `snippet_method` and `snippet_length` to the CSV.

4. **Write a short documentation README**  
   - Briefly describe what you did in each step (1–3) and how your code works on a high level.  
   - Explain why you chose specific models, features and methods.  
   - Mention at least one alternative model in step 1, 3 additional audio features for step 2 and one alternative audio thumbnailing method in step 3.  

**Submit your challenge**
   - Commit and push all the files to your private fork.
   - Invite `leonard.kinzinger@tum.de` and `benedikt.roder@tum.de` to the repository.
   - Respond to the coding task email and tell us how long it took you to complete the task and how difficult you (honestly) found it.
   - That’s it! :) Thank you for taking the time! We really appreciate the effort you put in here.

## About Replicate  

Replicate is a platform that hosts many state-of-the-art AI models across different modalities. All models can be accessed in a unified way via the **Replicate Python SDK**, which makes it easy to experiment with different approaches.  

We have sent you an **individual Replicate API key** in the coding task email. Your Replicate API key has a budget of **10 USD**. You can use this freely for this coding task.  

## Requirements  

- Use a Python virtual environment.  
- Keep your Replicate API key in a `.env` file (do not commit it).  
- Provide a `requirements.txt` with all dependencies.  
- Ensure reproducibility (someone else should be able to clone your repo, install dependencies, add their API key, and run everything).  
- Include a README as described above.  

We encourage you to use coding assistants. Just remember that you are responsible for the outputs.  

## What we will look at  

- **Correctness:** Does your pipeline run? Are audio files, snippets, and features saved?  
- **Code quality:** Structure, readability, error handling.  
- **Reproducibility:** Clear setup, requirements, .env handling.  
- **Documentation:** A clear README covering your choices.  
- **Creativity:** Interesting prompts, thoughtful features or alternative snippet ideas.  

## Getting Started  

1. Fork this repository and make it private.  

2. Clone your fork to your local machine:  
   ```bash
   git clone https://github.com/<your-username>/<your-private-fork>.git
   cd <your-private-fork>
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Add your Replicate API key to .env:
   ```bash
   REPLICATE_API_TOKEN=your_api_key_here
   ```

5. Run your pipeline scripts or notebooks.
   - You can authenticate with Replicate in two ways:
      - Option A – via environment variable:
      ```bash
      export REPLICATE_API_TOKEN=<your token>
      ```
      - Option B – via Python client:
      ```
      import os
      from replicate.client import Client

      replicate = Client(
          api_token=os.environ["REPLICATE_API_TOKEN"]
      )
      ```

That’s it and have fun! This task is about seeing how you work, not about perfection.
git test
