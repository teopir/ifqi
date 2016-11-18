"""
This file generates a document with all experiments.

"""
import json

def dicCompare(dic1, dic2, keyname):
    if isinstance(dic1, dict) and isinstance(dic2, dict):
        ret = []
        for key in dic1:
            ret += dicCompare(dic1[key], dic2[key], keyname + "/"+ key)
        return ret
    elif isinstance(dic1, dict) or isinstance(dic2, dict):
        return [(keyname, "dictionary", "not dictionary")]
    elif dic1!=dic2:
        return [(keyname, str(dic1), str(dic2))]
    else:
        return []
        
def strdic(dic, tab=0):
    ret = ""
    if isinstance(dic, dict):
        ret = "\t"*tab + "{\n"
        for key in dic:
            ret += "\t"*tab + key + ":" + strdic(dic[key],tab+1)
        ret += "\t"*tab + "}\n"
        return ret
    else:
        return str(dic)
        
def cleanJson(jsonFile, version=0):
    nIter = jsonFile["rlAlgorithm"]["nIterations"]
    if version==0:
        nExp = jsonFile["experimentSetting"]["experienceReplay"]
        if nExp>nIter:
            jsonFile["experimentSetting"]["experienceReplay"]=False
    else:
        if not "experienceReplay" in jsonFile["rlAlgorithm"]:
            jsonFile["rlAlgorithm"]["experienceReplay"] = False
        
def camelize(dic, key=False):
    if isinstance(dic,dict):
        ret = {}
        for key in dic:
            ret[camelize(key, key=True)] = camelize(dic[key])
        return ret
    elif key:
        ret = ""
        cam = False
        for letter in dic:
            if letter=="_":
                cam=True
            elif cam:
                ret+=letter.upper()
                cam=False
            else:
                ret+=letter
        return ret
    else:
        return dic

try:
    with open("diary.json", 'r') as fp:
        diary = json.load(fp)
except:
    diary = []

HTMLDoc = "<html>\n\t<head>\n\t\t<title>Experiments</title>\n\t</head>\n\t<body>"

tab=2
for exp in diary:
    
    importance = int(exp["importance"])
    stars = "&#9733"*importance + "&#9734"*(5-importance)
    HTMLDoc += "\t"*tab +  "<h1 name='" + exp["name"] + "'>" + exp["name"] + " " + stars + "</h1>\n"
    HTMLDoc += "\t"*tab +  "<h2>" + exp["date"] + "</h2>\n"
    HTMLDoc += "\t"*tab +  "<p><i>" + exp["description"] + "</i></p>\n"

    
    jsonFiles = exp["jsonFile"]


    jsonBase = jsonFiles[0]
    with open(jsonBase, 'r') as fp:
        jsonBase = json.load(fp)
        jsonBase = camelize(jsonBase)
        versionBase = 0
        if "version" in jsonBase:
            versionBase = jsonBase["version"]
        cleanJson(jsonBase,version=versionBase)



    if versionBase==0:
        models = jsonBase["model"]
    else:
        models = jsonBase["regressors"]

    mdp = jsonBase["mdp"]
    setting = jsonBase["experimentSetting"]
    rl_setting = jsonBase["rlAlgorithm"]
    #nDataset
    #nIter
    #experienceReplay
    
    HTMLDoc += "\t"*tab + "<h4>Base Parameters</h4>\n"

    if versionBase==0:
        HTMLDoc += "\t" * tab + "<h5>Model</h5>\n"
        HTMLDoc += "\t" * tab + "<p><font face='courier' size='2'>" + str(models) + "</font></p>\n"
    else:
        HTMLDoc += "\t"*tab +  "<h5>Model("+ str(len(models)) + ")</h5>\n"
        for m in models:
            HTMLDoc += "\t"*tab +  "<p><font face='courier' size='2'>" + str(m) + "</font></p>\n"
    
    HTMLDoc += "\t"*tab +  "<h5>Experiment Settings</h5>\n"
    HTMLDoc += "\t"*tab +  "<p><font face='courier' size='2'>" + str(setting) + "</font></p>\n"
    
    HTMLDoc += "\t"*tab +  "<h5>FQI setting</h5>\n"
    HTMLDoc += "\t"*tab +  "<p><font face='courier' size='2'>" + str(rl_setting) + "</font></p>\n"
    
    HTMLDoc += "\t"*tab +  "<h5>Environment</h5>\n"
    HTMLDoc += "\t"*tab +  "<p><font face='courier' size='2'>" + str(mdp) + "</font></p>\n"
    
    for jsonPath in jsonFiles:
        try:
            with open(jsonPath, 'r') as fp:
                jsonFile = json.load(fp)
                jsonFile = camelize(jsonFile)
                cleanJson(jsonFile,versionBase)
                
        except:
            raise Exception("Json not found")
            
        diff = dicCompare(jsonBase, jsonFile, "")
        if len(diff) > 0:
            tab+=1
            HTMLDoc += "\t"*tab +  "<h4>Differeces</h4>\n"
            
            HTMLDoc += "\t"*tab +  "<p>"            
            for d in diff:
                HTMLDoc += "\t"*(tab+1) +  "<font face='courier' size='2' color='red'>" + d[0] + " is " + d[2] + " instead of " + d[1] + "</font><br>\n"
            
            HTMLDoc += "\t"*tab +  "</p>"     
            HTMLDoc += "\t"*tab + "<br/>"
            tab -=1
            
    for image in exp["images"]:
        HTMLDoc += "\t"*tab +  "<h4>" + image["title"] +"</h4>\n"
        HTMLDoc += "\t"*tab +  "<img src='" + image["dir"] +"'></img>\n"
        HTMLDoc += "\t"*tab +  "<p>" + image["description"] + "<p/>\n"
            
    HTMLDoc += "\t"*tab +  "<hr/>" 
    
HTMLDoc += "\n\t</body></html>"

with open("diary.html", 'w') as fp:
    fp.write(HTMLDoc)