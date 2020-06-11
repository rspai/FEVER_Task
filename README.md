# FEVER_Task
Fact Extraction and Verification

The Shared FEVER task is to build a system that can retrieve evidence from a corpus of Wikipedia summaries for a given claim and verify the authenticity of the same. 
We propose a four layered pipeline with:
  “Document Retrieval”, 
  “Sentence Selection”, 
  “Textual Entailment”, and 
  “Label Prediction”. 
The Wikipedia corpus was indexed using Apache Solr. 
Named Entity Recognition was used to enhance the performance of the Document Retrieval stage. 
Sentence similarity was determined using Universal Sentence Encoder. 
Decomposable Attention Model for textual entailment was adapted for the FEVER dataset. 
Finally, the label was predicted based on the weighted scores calculated for the claim evidence pairs. 

Our approach achieved a label accuracy of 0.5843, Fever Score of 0.3741 and F1 score of 0.2512.

