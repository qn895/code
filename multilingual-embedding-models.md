## Lost In Translation? Multilingual Embedding Models Are All You Need

### Introduction

In a world of global users, cross-lingual informational retrieval (CLIR) is crucial. Instead of limiting searches to a single language, CLIR lets you find information in any language, enhancing user experience and streamlining operations. Imagine a global market where ecommerce customers can search for items in their language, and the right results would pop up, without the need for localizing the data ahead of time. Or, where academic researchers can search for papers in their native language, with nuance and complexity, even if the source is in another. 

Multilingual text embedding models allow us to do just that. Embeddings are a way to represent the meaning of text as numerical vectors. These vectors are designed so that texts with similar meanings are located close to each other in a high-dimensional space. Multilingual text embedding models specifically are designed to map words and phrases with the same meaning across different languages into a similar vector space.

Models like the open-source Multilingual E5 are trained on massive amounts of text data, often using techniques like contrastive learning. In this approach, the model learns to distinguish between pairs of texts with similar meanings (positive pairs) and those with dissimilar meanings (negative pairs). The model is trained to adjust the vectors it produces such that the similarity between positive pairs is maximized, and the similarity between negative pairs is minimized. For multilingual models, this training data includes text pairs in different languages that are translations of each other, enabling the model to learn a shared representation space for multiple languages. The resulting embeddings can then be used for various NLP tasks, including cross-lingual search, where the similarity between text embeddings is used to find relevant documents regardless of the language of the query.


### Benefits of Multilingual Vector Search

**Nuance**: Vector search excels at capturing semantic meaning, going beyond keyword matching. This is crucial for tasks that require understanding context and subtleties in language.

**Cross-Lingual Understanding**: Enables effective information retrieval across languages, even when the query and documents use different vocabulary.

**Relevance**: Delivers more relevant results by focusing on the conceptual similarity between queries and documents.

For example, consider an academic researcher studying the "impact of social media on political discourse" across different countries. With vector search, they can input queries like "l'impatto dei social media sul discorso politico" (Italian) or "ảnh hưởng của mạng xã hội đối với diễn ngôn chính trị" (Vietnamese) and find relevant papers in English, Spanish, or any other indexed language. This is because vector search identifies papers that discuss the concept of social media's influence on politics, not just those containing the exact keywords. This greatly enhances the breadth and depth of their research.

### Getting Started

Here's how to set up CLIR using Elasticsearch - with the E5 model that is provided out of the box. We’ll use the open-source multilingual COCO dataset, which contains image captions in multiple languages to help us visualize 2 types of searches: 
- Queries and search terms in other languages on one English dataset, and 
- Queries in multiple languages on top of a dataset containing documents in multiple languages. 

Then, we will leverage the power of hybrid search and reranking to improve the search results even further.

**Prerequisites**
- Python 3.6+
- Elasticsearch 8+
- Elasticsearch Python client: pip install elasticsearch

#### Dataset
The COCO dataset is a large-scale captioning dataset. Each image in the dataset is captioned in multiple different languages, with several translations available per language. For demonstration purposes, we’ll index each translation as an individual document, along with the first available English translation for reference.

**Step 1: Download the Multilingual COCO Dataset**

To simplify the blog and make it easier to follow along, here we are loading the first 100 rows of the restval into a local JSON file with a simple API call. Alternatively, you can use HuggingFace’s library datasets to load the full dataset, or subsets of the dataset.   

```python
import requests
import json
import os
url = "https://datasets-server.huggingface.co/rows?dataset=romrawinjp%2Fmultilingual-coco&config=default&split=restval&offset=0&length=100"
response = requests.get(url)


if response.status_code == 200:
   data = response.json()
   output_file = "multilingual_coco_sample.json"
   with open(output_file, "w", encoding="utf-8") as f:
       json.dump(data, f, indent=4, ensure_ascii=False)
   print(f"Data successfully downloaded and saved to {output_file}")
else:
   print(f"Failed to download data: {response.status_code}")
   print(response.text)
```


If the data is loaded  into a JSON file successfully you should see something similar to the following:
```
Data successfully downloaded and saved to multilingual_coco_sample.json
```

**Step 2: (Start Elasticsearch) and Index the Data in Elasticsearch**

Start your local or [cloud Elasticsearch server](https://cloud.elastic.co/registration?onboarding_token=vectorsearch&cta=cloud-registration&tech=trial&plcmt=article%20content&pg=search-labs). Then, initiate the Elasticsearch client and index the data. 

```python
from elasticsearch import Elasticsearch
from getpass import getpass
import json


# Initialize Elasticsearch client
es = Elasticsearch(getpass("Host: "), api_key=getpass("API Key: "))


index_name = "coco"


# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
   es.indices.create(index=index_name, body=mapping)


# Load the JSON data
with open('./multilingual_coco_sample.json', 'r') as f:
   data = json.load(f)


rows = data["rows"]
# List of languages to process
languages = ["en", "es", "de", "it", "vi", "th"]


bulk_data = []
for data in rows:
   row = data["row"]
   image = row.get("image")
   image_url = image["src"]


   # Process each language
   for lang in languages:
       # Skip if language not present in this row
       if lang not in row:
           continue


       # Get all descriptions for this language
 # along with first available English caption for reference
       descriptions = row[lang]
       first_eng_caption = row["en"][0]


       # Prepare bulk indexing data
       for description in descriptions:
           if description == "":
               continue
           # Add index operation
           bulk_data.append(
               {"index": {"_index": index_name}}
           )
           # Add document
           bulk_data.append({
               "language": lang,
               "description": description,
               "en": first_eng_caption,
               "image_url": image_url,
           })


# Perform bulk indexing
if bulk_data:
   try:
       response = es.bulk(operations=bulk_data)
       if response["errors"]:
           print("Some documents failed to index")
       else:
           print(f"Successfully bulk indexed {len(bulk_data)} documents")
   except Exception as e:
       print(f"Error during bulk indexing: {str(e)}")


print("Indexing complete!")
```


Once the data is indexed, you should see something similar to the following:
```
Successfully bulk indexed 4840 documents
Indexing complete!
```

**Step 3: Deploy E5 trained model**

In Kibana, navigate to the **Stack Management** > **Trained Models** page, and click Deploy for the `.multilingual-e5-small_linux-x86_64` option. This E5 model is a small multilingual optimized for linux-x86_64, which we can use out of the box for. 

Clicking on **Deploy** will show a screen where you can adjust the deployment settings, or vCPUs configurations. For simplicity, we will go with the default options, with adaptive resources selected which will auto-scale our deployment depending on usage.



Optionally, if you want to use other text embedding models, you can. For example, to use the BGE-M3, you can use Elastic’s Eland Python client to import the model from HuggingFace. 
```bash
export MODEL_ID="bge-m3"
export HUB_MODEL_ID="BAAI/bge-m3"
export CLOUD_ID={{CLOUD_ID}}
export ES_API_KEY={{API_KEY}}
docker run -it --rm docker.elastic.co/eland/eland \
eland_import_hub_model --cloud-id $CLOUD_ID --es-api-key $ES_API_KEY --hub-model-id $HUB_MODEL_ID --es-model-id $MODEL_ID --task-type text_embedding --start
```

Then, navigate to the Trained Models page to deploy the imported model with the desired configurations.

**Step 4: Vectorize or create embeddings for the original data with the deployed model
**

To create the embeddings, we first need to create an ingest pipeline that will allow us to take the text and run it through the inference text embedding model. You can do this in the Kibana UI or through Elasticsearch’s API.

**To do via the Kibana interface**, after deploying the Trained Model, click the **Test** button. This will give you the ability to test and preview the generated embeddings. Create a new data view for the coco index, set Data view to the newly created coco data view, and set the Field to description because that’s the field we want to generate embeddings for.

That works great! Now we can proceed to create the ingest pipeline and reindex our original documents, pass it through the pipeline, and create a new index with the embeddings. You can achieve this by clicking Create pipeline, which will guide you through the pipeline creation process, with auto-populated processors needed to help you create the embeddings.

The wizard can also auto-populate the processors needed to handle failures while ingesting and processing the data.
Let’s now create the ingest pipeline. I’m naming the pipeline coco_e5. Once the pipeline is created successfully, you can immediately use the pipeline to generate the embeddings by reindexing the original indexed data to a new index in the wizard. Click Reindex to start the process.


**For more complex configurations, we can utilize the Elasticsearch API. **

For some models, because of the way the models were trained, we might need to prepend or append certain texts to the actual input before generating the embeddings, otherwise we will see a performance degradation. 
For example, with the e5, the model expects the input text to follow “passage: {content of passage}”. Let’s utilize the ingest pipelines to accomplish that: We’ll create a new ingest pipeline vectorize_descriptions. In this pipeline, we will create a new temporary temp_desc field, prepend “passage: “ to the description text, run temp_desc  through the model to generate text embeddings, and then delete the temp_desc.

```
PUT _ingest/pipeline/vectorize_descriptions
{
"description": "Pipeline to run the descriptions text_field through our inference text embedding model",
"processors": [
 {
   "set": {
     "field": "temp_desc",
     "value": "passage: {{description}}"
   }
 },
 {
   "inference": {     
"field_map": {
       "temp_desc": "text_field"
     },
     "model_id": ".multilingual-e5-small_linux-x86_64_search",
     "target_field": "vector_description"
   }
 },
 {
   "remove": {
     "field": "temp_desc"
   }
 }
]
}
```
In addition, we might want to specify what type of quantization we want to use for the generated vector. By default, Elasticsearch uses int8_hnsw, but here I want Better Binary Quantization (or bqq_hnsw) which reduces each dimension to a single bit precision. This reduces the memory footprint by 96% (or 32x) at a larger cost of accuracy. I’m opting for this quantization type because I know I’ll use a reranker later on to improve the accuracy loss. 

To do that, we will create a new index named coco_multi, and specify the mappings. The magic here is in the vector_description field, where we specify the index_options’s type to be bbq_hnsw.
```
PUT coco_multi
{
 "mappings": {
   "properties": {
     "description": {
       "type": "text"
     },
     "en": {
       "type": "text"
     },
     "image_url": {
       "type": "keyword"
     },
     "language": {
       "type": "keyword"
     },
     "vector_description.predicted_value": {
       "type": "dense_vector",
       "dims": 384,
       "index": "true",
       "similarity": "cosine",
       "index_options": {
         "type": "bbq_hnsw" 
       }
     }
   }
 }
}
```


Now, we can reindex the original documents to a new index, with our ingest pipeline that will “vectorize” or create embeddings for the descriptions field.
```
POST _reindex?wait_for_completion=false
{
 "source": {
   "index": "coco"
 },
 "dest": {
   "index": "coco_multilingual",
   "pipeline": "vectorize_descriptions"
 }
}
```


Great, after the task is done running, performing the search will give us documents in multiple languages, with the “en” field for us to reference:
```
 # GET coco_multilingual/_search
    {
       "_index": "coco_multilingual",
       "_id": "WAiXQJYBgf6odR9bLohZ",
       "_score": 1,
       "_source": {
         "description": "Ein Parkmeßgerät auf einer Straße mit Autos",
         "en": "A row of parked cars sitting next to parking meters.",
         "language": "de",
         "vector_description": {...}
       }
     },
     . . .
```
Let’s try now to perform the search in English and see how well it does:
```
GET coco_multi/_search
{
"size": 10,
"_source": [
  "description", "language", "en"
],
"knn": {
  "field": "vector_description.predicted_value",
  "k": 10,
  "num_candidates": 100,
  "query_vector_builder": {
    "text_embedding": {
      "model_id": ".multilingual-e5-small_linux-x86_64_search",
      "model_text": "query: kitty"
    }
  }
}
}
```
Results are:
```
     {
       "_index": "coco_multi",
       "_id": "JQiXQJYBgf6odR9b6Yz0",
       "_score": 0.9334303,
       "_source": {
         "description": "Eine Katze, die in einem kleinen, gepackten Koffer sitzt.",
         "en": "A brown and white cat is in a suitcase.",
         "language": "de"
       }
     },
      {
       "_index": "coco_multi",
       "_id": "3AiXQJYBgf6odR9bFod6",
       "_score": 0.9281012,
       "_source": {
         "description": "Una bambina che tiene un gattino vicino a una recinzione blu.",
         "en": "A little girl holding a kitten next to a blue fence.",
         "language": "it"
       }
     },
     . . .
```

Here, even though the query looks deceptively simple, we are searching for the numerical embeddings of the word ‘kitty’ across all documents in all languages underneath the hood. And because we are performing vector search, we are able to semantically search for all words that might be related to ‘kitty’: “cat”, “kitten”, “feline”, “gatto” (Italian), “mèo” (Vietnamese), 고양이 (Korean), 猫 (Chinese), etc. As a result, even if my query is in English, we can search for content in all other languages too. For example, searching a kitty lying on something gives back documents in Italian, Dutch, or Vietnamese too. Talk about efficiency! 

```
GET coco_multi/_search
{  
 "size": 100,
 "_source": [
   "description", "language", "en"
 ],
 "knn": {
   "field": "vector_description.predicted_value",
   "k": 50,
   "num_candidates": 1000,
   "query_vector_builder": {
     "text_embedding": {
       "model_id": ".multilingual-e5-small_linux-x86_64_search",
       "model_text": "query: kitty lying on something"
     }
   }
 }
}
```
Would give us:
```
{
 "description": "A black kitten lays on her side beside remote controls.",
 "en": "A black kitten lays on her side beside remote controls.",
 "language": "en"
},
{
 "description": "un gattino sdraiato su un letto accanto ad alcuni telefoni ",
 "en": "A black kitten lays on her side beside remote controls.",
 "language": "it"
},
{
 "description": "eine Katze legt sich auf ein ausgestopftes Tier",
 "en": "a cat lays down on a stuffed animal",
 "language": "de"
},
{
 "description": "Một chú mèo con màu đen nằm nghiêng bên cạnh điều khiển từ xa.",
 "en": "A black kitten lays on her side beside remote controls.",
 "language": "vi"
}
. . .
```

Similarly, performing keyword search for “cat” in Korean (“고양이”), will also give back meaningful results. What’s spectacular here is that we don’t even have any documents in Korean in this index!
```
GET coco_multi/_search
{
 "size": 100,
 "_source": [
   "description", "language", "en"
 ],
 "knn": {
   "field": "vector_description.predicted_value",
   "k": 50,
   "num_candidates": 1000,
   "query_vector_builder": {
     "text_embedding": {
       "model_id": ".multilingual-e5-small_linux-x86_64_search",
       "model_text": "query: 고양이"
     }
   }
 }
}
```

```
     {
       {
         "description": "eine Katze legt sich auf ein ausgestopftes Tier",
         "en": "a cat lays down on a stuffed animal",
         "language": "de"
       }
     },
     {
       {
         "description": "Một con chó và con mèo đang ngủ với nhau trên một chiếc ghế dài màu cam.",
         "en": "A dog and cat lying  together on an orange couch. ",
         "language": "vi"
       }
     },
```

This works because the embedding model represents meaning in a shared semantic space, allowing retrieval of relevant images even with a query in a different language than the indexed captions.

**Step 5: Increase relevant search results with hybrid search and reranking **

We are happy that the relevant results showed up as expected. But, in the real world, say in ecommerce or in RAG applications that need to narrow down to the top 5-10 most applicable results, we can use a rerank model to prioritize the most relevant results. 
Here, performing a query that asks “what color is the cat?” in Vietnamese will yield a lot of results, but the top 1 or 2 might not be the most relevant. 

```
GET coco_multi/_search
{
 "size": 20,
 "_source": [
   "description",
   "language",
   "en"
 ],
 "knn": {
   "field": "vector_description.predicted_value",
   "k": 20,
   "num_candidates": 1000,
   "query_vector_builder": {
     "text_embedding": {
       "model_id": ".multilingual-e5-small_linux-x86_64_search",
       "model_text": "query: con mèo màu gì?"
     }
   }
 }
}
```
The results all mention cat, or some form of color in the sentence. For example, the `orange couch`.
```
{
        "_index": "coco_multi",
        "_id": "4giXQJYBgf6odR9b3Iu8",
        "_score": 0.93522114,
        "_source": {
          "description": "Một con chó và con mèo nằm cùng nhau trên một chiếc ghế dài màu cam.",
          "en": "A dog and cat lying  together on an orange couch. ",
          "language": "vi"
        }
      },
      {
        "_index": "coco_multi",
        "_id": "qwiYQJYBgf6odR9bBYwQ",
        "_score": 0.933766,
        "_source": {
          "description": "Một chú mèo con màu đen nằm cạnh hai cái điều khiển từ xa.",
          "en": "A black kitten lays on her side beside remote controls.",
          "language": "vi"
        }
      }
```
But we want to know what color is the cat, so let’s improve that!  Let’s integrate [Cohere](https://cohere.com/blog/rerank-3pt5)’s multilingual rerank model to improve the reasoning corresponding to our question.

```
PUT _inference/rerank/cohere_rerank
{
 "service": "cohere",
 "service_settings": {
   "api_key": "your_api_key",
   "model_id": "rerank-v3.5"
 },
 "task_settings": {
   "top_n": 10,
   "return_documents": true
 }
}
```

```
GET coco_multi/_search
{
"size": 10,
"_source": [
  "description",
  "language",
  "en"
],
"retriever": {
  "text_similarity_reranker": {
    "retriever": {
      "rrf": {
        "retrievers": [
          {
            "knn": {
              "field": "vector_description.predicted_value",
              "k": 50,
              "num_candidates": 100,
              "query_vector_builder": {
                "text_embedding": {
                  "model_id": ".multilingual-e5-small_linux-x86_64_search",
                  "model_text": "query: con mèo màu gì?" // English: What color is the cat?
                }
              }
            }
          }
        ],
        "rank_window_size": 100,
        "rank_constant": 0
      }
    },
    "field": "description",
    "inference_id": "cohere_rerank",
    "inference_text": "con mèo màu gì?"
  }
}
}
```
```
     {
       "_index": "coco_multi",
       "_id": "rQiYQJYBgf6odR9bBYyH",
       "_score": 1.5501487,
       "_source": {
         "description": "Hai cái điện thoại được đặt trên một cái chăn cạnh một con mèo con màu đen.",
         "en": "A black kitten lays on her side beside remote controls.",
         "language": "vi"
       }
     },
     {
       "_index": "coco_multi",
       "_id": "swiXQJYBgf6odR9b04uf",
       "_score": 1.5427427,
       "_source": {
         "description": "Một con mèo sọc nâu nhìn vào máy quay.", // Real translation: A brown striped cat looks at the camera 
         "en": "This cat is sitting on a porch near a tire.",
         "language": "vi"
       }
     },
```
Now, with the top results, our application can confidently answer that the kitten’s color is black, or brown with stripes. What’s even more interesting here is our vector search actually caught an omission in the English caption in the original dataset. It’s able to find the brown striped cat even though the reference English translation missed that detail. This is the power of vector search.

### Conclusion
In this blog, we have walked through the utility of a multilingual embedding model, and how to leverage Elasticsearch to integrate the models to generate embeddings, and to effectively improve relevance and accuracy with hybrid search and reranker.  You can create a Cloud cluster of your own to try multilingual semantic search using our out-of-the-box E5 model on the language and dataset of your choice.



