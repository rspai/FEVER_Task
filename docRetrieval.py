from urllib.request import urlopen
import urllib.parse
import urllib.request
import json
from time import time
from urllib.error import HTTPError,URLError

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()
total_time=0
start = time()
data = []
with open('../Datasets/shared_task_test.jsonl') as f:
#with open('../Datasets/docRetrieval/zeroRecallDocs.json') as f:
    #nf = open("../Datasets/docRetrieval/rePrunedDocRetrieval.json","a")
    count_lines = 1
    qtTotal=0
    for line in f:
        
        try:
            t1 = time()
            claim = json.loads(line)
            q = claim['claim']
            ners = nlp(q)

            q=q.replace('\n','')
            q=q.replace(':',' ')
            q=q.replace(';',' ')
            q=q.replace('\'','')
            q=q.replace('\"','')
            
            bq=""
            for e in ners.ents:
                #print("entity:",e.text)
                if e.label_ not in ["DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"]:
                    bq+="id_tfidf:("+e.text+")^20 OR "  
            #print(bq)
            bq = bq.rstrip(" OR ")        
            #print(bq)
            bq_query=""
            if bq != "":
                bq_query="&bq="+urllib.parse.quote(bq)
            else:
                bq="id_tfidf:("+claim['claim']+")^20"
                bq_query="&bq="+urllib.parse.quote(bq)
            #print(bq_query)

            query =  urllib.parse.quote(q)
            #inurl = "http://3.86.163.51:8983/solr/wiki/select?q="+query+"&rows=20" 
            qt1 = time()
            inurl = "http://localhost:8983/solr/wiki/select?defType=dismax&fl=id%2Ctext&pf=text_tfidf%5E10%20id_tfidf%5E20&rows=10&q="+query+bq_query
            qt2 = time()
            connection = urlopen(inurl).read().decode('utf-8')
            qtTotal += (qt2-qt1)
            response = json.loads(connection)
            docPacket = {}
            docPacket["id"]=claim["id"]
            docPacket["claim"] = claim['claim']
            #docPacket["label"] = claim["label"]
            #docPacket["evidence"] = claim["evidence"]
            docPacket["docs"] = [ {"doc_id":t["id"],"text":t["text"]} for t in response['response']['docs']]
            docJson = json.dumps(docPacket)
            nf = open("../Datasets/docRetrieval/test/doc_"+str(count_lines//2000) +".json","a")
            nf.write(docJson+'\n')
            nf.close()
            t2= time()
            time_diff = t2-t1
            total_time +=time_diff
            count_lines+=1
            if(count_lines%1000==0):
                print("Average query response time:"+ str(qtTotal/count_lines))
                print("average time per line:"+str(total_time/count_lines))
    
        except HTTPError as err:
            eFile = open("docretrieval_error.log",'a')
            errPacket ={"claim":q,"query":inurl,"error_code":err.code}
            print("Http Error Occured Check Logs:"+q)
            eFile.write(str(errPacket)+'\n')
            eFile.close()
            
        except ValueError as err:
            eFile = open("docretrieval_error.log",'a')
            errPacket ={"claim":q,"query":inurl,"error_code":err}
            print("Some ValueError Occured Check Logs:"+q)
            eFile.write(str(errPacket)+'\n')
            eFile.close()
#nf.close()
end = time()
print(end-start)