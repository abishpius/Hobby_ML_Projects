Link to Preview for Free: https://colab.research.google.com/drive/1HfPEl8evpDPRPdXuTOyv_nH-1SmzNwdF?usp=sharing
![image](https://github.com/user-attachments/assets/aad6a1cf-c8cf-448f-a22e-2e01b5145072)


## Inspiration
I am a professional Data Scientist and wanted to test out creating a custom NL to SQL Agent that could integrate with MongoDB through PyMongo as I was scouring the blogs and seeing that it was not something that was readily available. Instead there were pages and pages on RAG and vector search, so I thought why not have something simple that an analyst or low code tool user could readily access to get answers about data instead of spending hours waiting for an email response from a Data Scientist (yes, I am guilty of long response times 😛 ).

## What it does
It is an agent that mimics a Data Scientist's role on the fundamental level by answering questions about and providing visualizations for any MongoDB dataset.

## How we built it
It is built in python using PyMongo to connect to the MongoDB database. It utilizes Gemini api as the main LLM and Langchain to set up the agentic workflow. The agent has access to four tools , three custom ones for interacting with MongoDBs and one python repl for plotting. The UI is created using Google's Mesop python framework designed for ready purpose LLM apps. It is deployed on Google Cloud Run.

## Challenges we ran into
Prompt engineering and memory were the two most frustrating. Preserving the memory buffer in the state of the UI was more challenging than expected especially when layering components like I did. Prompt engineering proved interesting, maybe it is the difference between using Flash vs thinking Gemini models, as it seemed to not work well with either the popular ReAct nor COT prompt patterns.

## Accomplishments that we're proud of
The mesop UI (adding Dark Mode specifically was the best ☺️) and the fact it can render plots effectively!

## What we learned
Got to whitelist your IP to access MongoDB databases 😂. App prototyping with Mesop.

## What's next for PyMon Go! The Data Science MongoDB Agent
Next steps, I would love to submit a blog article to be featured on the MongoDB website with this example, I feel its useful and can offer a good amount of insight.
I would like to also resolve a handful of niche errors like:
- Adding error handling logic for various agent faced exceptions
- Adding support/tool for retrieving from across multiple tables
- Adding nuance for handling non-numeric data columns
