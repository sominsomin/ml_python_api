from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import sys
# from flask_cors import CORS

#MODELS_PATH_RENT = '/home/simon/Documents/real_estate_and_ai/analysis/exported_model/wohnung_mieten'
#MODELS_PATH_BUY = '/home/simon/Documents/real_estate_and_ai/analysis/exported_model/wohnung_kaufen'

MODELS_PATH_BUY = '/home/ubuntu/ml_python_api/models/wohnung_kaufen'
MODELS_PATH_RENT = '/home/ubuntu/ml_python_api/models/wohnung_mieten'

#MODELS_PATH = 'models/'

SCALERX_FILENAME = 'scalerX.model'
SCALERY_FILENAME = 'scalerY.model'
MODEL_FILENAME = 'model.model'
COLUMN_FILENAME = 'columns.txt'
COLUMNDF_FILENAME = 'columnsDf.txt'

TYPE_BUY = 'buy'
TYPE_RENT = 'rent'

FIELDS_wohnung_kaufen = {
        'kiez' : 
            {'type' : 'nominal',
             'order' : 1,
             'display_Name' : 'area',
             'unit' : '',
             'values' : ['Köpenick', 'Mitte', 'Prenzlauer Berg', 'Kreuzberg',
                         'Charlottenburg', 'Spandau', 'Wilmersdorf', 'Hohenschönhausen',
                         'Steglitz', 'Wedding', 'Lichtenberg', 'Weißensee', 'Hellersdorf',
                         'Zehlendorf', 'Tempelhof', 'Pankow', 'Tiergarten', 'Schöneberg',
                         'Treptow', 'Friedrichshain', 'Reinickendorf', 'Neukölln',
                         'Marzahn']
             },
        'wohnflaeche' : 
            {'type' : 'float',
             'order' : 2,
             'display_Name' : 'living space',
             'unit' : '[m²]',
             'value_min' : 20, 
             'value_max' : 350},
        'baujahr' : 
            {'type' : 'int', 
             'order' : 3,
             'display_Name' : 'construction year',
             'unit' : '',
             'value_min' : 1900, 
             'value_max' : 2020},
        'etage' : 
            {'type' : 'int', 
             'order' : 4,
             'display_Name' : 'level',
             'unit' : '',
             'value_min' : -1, 
             'value_max' : 12},
        'personenaufzug' : 
            {'type' : 'boolean', 
             'order' : 4,
             'display_Name' : 'elevator',
             'unit' : '',
             'values' : ['no', 'yes',],
             },
        # 'energieeffizienzklasse' : 
        #     {'type' : 'nominal', 
        #       'order' : 4,
        #       'display_Name' : 'Energieeffizienzklasse',
        #       'unit' : '',
        #       'values' : [ 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        #       },
        'einbaukueche' :
            {'type' : 'boolean', 
             'order' : 5,
             'display_Name' : 'built-in kitchen',
             'unit' : '',
             'values' : ['no', 'yes',],
             'value_min' : 0, 
             'value_max' : 1},
        'garten_mitbenutzung' :
            {'type' : 'boolean', 
              'order' : 5,
              'display_Name' : 'garden sharing',
              'unit' : '',
              'values' : ['no', 'yes',],
              'value_min' : 0, 
              'value_max' : 1},
        'gaeste_wc' :
            {'type' : 'boolean', 
             'order' : 6,
             'display_Name' : 'guest toilet',
             'unit' : '',
             'values' : ['no', 'yes',],
             'value_min' : 0, 
             'value_max' : 1},
        'balkon_terasse'  :
            {'type' : 'boolean', 
             'order' : 7,
             'display_Name' : 'balcony',
             'unit' : '',
             'values' : ['no', 'yes',],
             'value_min' : 0, 
             'value_max' : 1}, 
        'ausstattung' : 
            {'type' : 'nominal', 
             'order' : 8,
              'display_Name' : 'equipment',
              'unit' : '',
              'values' : ['Zentralheizung', 'Fernwärme', 'Fußbodenheizung',
                          'Etagenheizung', 'Gas-Heizung', 'Öl-Heizung',
                          'sonstige'
                  ],
              },
        'objektzustand' :
            {'type' : 'nominal',
             'order' : 9,
              'display_Name' : 'object state',
              'unit' : '',
              'values' : ['Erstbezug' , 'Gepflegt', 'Neuwertig', 'Modernisiert',
                          'Erstbezug nach Sanierung', 'Saniert', 'Vollständig renoviert',
                          'Renovierungsbedürftig']
              }, 
        }
    
    
FIELDS_wohnung_mieten = {
        'kiez' : 
            {'type' : 'nominal',
             'order' : 1,
             'display_Name' : 'area',
             'unit' : '',
             'values' : ['Köpenick', 'Mitte', 'Prenzlauer Berg', 'Kreuzberg',
                         'Charlottenburg', 'Spandau', 'Wilmersdorf', 'Hohenschönhausen',
                         'Steglitz', 'Wedding', 'Lichtenberg', 'Weißensee', 'Hellersdorf',
                         'Zehlendorf', 'Tempelhof', 'Pankow', 'Tiergarten', 'Schöneberg',
                         'Treptow', 'Friedrichshain', 'Reinickendorf', 'Neukölln',
                         'Marzahn']
             },
        'wohnflaeche' : 
            {'type' : 'float',
             'order' : 2,
             'display_Name' : 'living space',
             'unit' : '[m²]',
             'value_min' : 20, 
             'value_max' : 350},
        'baujahr' : 
            {'type' : 'int', 
             'order' : 3,
             'display_Name' : 'construction year',
             'unit' : '',
             'value_min' : 1900, 
             'value_max' : 2020},
        'etage' : 
            {'type' : 'int', 
             'order' : 4,
             'display_Name' : 'level',
             'unit' : '',
             'value_min' : -1, 
             'value_max' : 12},
        'balkon_terasse' : 
            {'type' : 'boolean', 
             'order' : 4,
             'display_Name' : 'balcony',
             'unit' : '',
             'values' : ['no', 'yes',],
             },
        'keller' : 
            {'type' : 'boolean', 
              'order' : 4,
              'display_Name' : 'basement',
              'unit' : '',
              'values' : ['no', 'yes',],
              },
        'gaeste_wc' :
            {'type' : 'boolean', 
             'order' : 6,
             'display_Name' : 'guest toilet',
             'unit' : '',
             'values' : ['no', 'yes',],
             'value_min' : 0, 
             'value_max' : 1},
        'ausstattung' : 
            {'type' : 'nominal', 
             'order' : 8,
              'display_Name' : 'equipment',
              'unit' : '',
              'values' : ['Zentralheizung', 'Fernwärme', 'Fußbodenheizung',
                          'Etagenheizung', 'Gas-Heizung', 'Öl-Heizung',
                          'sonstige'
                  ],
              },
        'objektzustand' :
            {'type' : 'nominal',
             'order' : 9,
              'display_Name' : 'object state',
              'unit' : '',
              'values' : ['Erstbezug' , 'Gepflegt', 'Neuwertig', 'Modernisiert',
                          'Erstbezug nach Sanierung', 'Saniert', 'Vollständig renoviert',
                          'Renovierungsbedürftig']
              }, 
        }
    
class RealEstatePredictor :
    '''
    '''
    
    scalerX = ''
    scalerY = ''
    model = ''
    columns = ''
    columnsDf = ''
    pathToFiles = ''
    fields = ''
    
    def __init__(self, pathToFiles_, fields_) :
        ''''''
        self.pathToFiles = pathToFiles_
       # self.scalerX = joblib.load('{}/{}'.format(self.pathToFiles,SCALERX_FILENAME))
       # self.scalerY = joblib.load('{}/{}'.format(self.pathToFiles,SCALERY_FILENAME))
       # self.model = joblib.load('{}/{}'.format(self.pathToFiles,MODEL_FILENAME))
       # self.columns = joblib.load('{}/{}'.format(self.pathToFiles,COLUMN_FILENAME))
       # self.columnsDf = joblib.load('{}/{}'.format(self.pathToFiles,COLUMNDF_FILENAME))
        self.fields = fields_
                
    def predictValueFromDict(self, dataDict) :
    
        dictAsDf = self.preProcessDataDict(dataDict)
        
        transformedDict = self.scalerX.transform(dictAsDf)
            
        df = pd.DataFrame(transformedDict)
        
        prediction = self.scalerY.inverse_transform(pd.DataFrame(self.model.predict(df)))
        
        return prediction

    def preProcessDataDict(self, dataDict) :
        
        for key, field in self.fields.items() :
            if key in dataDict : 
                if field['type'] == 'float' : 
                    value = float(dataDict[key])
                    dataDict[key] = value
                elif field['type'] == 'int' : 
                    value = int(dataDict[key])
                    dataDict[key] = value
                if field['type'] == 'boolean' :
                    if dataDict[key] == 'no' : 
                        dataDict[key] = 0
                    else :
                        dataDict[key] = 1  
                        
        for key in dataDict : 
            if isinstance(dataDict[key],str) :
                dataDict[key] = dataDict[key].replace(' ','')
                        
        df = pd.DataFrame(dataDict, columns=self.columnsDf, index=[0])
        df_encoded = pd.get_dummies(df)
        df_withAllColumns = pd.DataFrame(df_encoded, columns=self.columns)
        df_withAllColumns = df_withAllColumns.fillna(0)
    
        return df_withAllColumns
    
    def checkValues(self,dataDict) :
    
        response = ''
        
        for key, field in self.fields.items() :
            if key in dataDict :
                value_type = field['type']
                if value_type == 'float' or value_type== 'int' :
                    value = float(dataDict[key])
                    value_min = valueToType(field['value_min'], value_type)
                    value_max = valueToType(field['value_max'], value_type)
                    if value > value_max or value < value_min :
                        return False, 'value not in range'
                        break
                    
        return True, response
    

application = Flask(__name__)
application.config['JSON_SORT_KEYS'] = False
# CORS(application)

@application.route('/getPrediction', methods=['POST'])
def getPrediction():
    
    if rentModel and buyModel:
        try:
            if request.is_json :
                json = request.get_json()
                
                print(json)
                
                if json['type'] == TYPE_RENT :
                    model = rentModel
                elif json['type'] == TYPE_BUY :
                    model = buyModel
                else : 
                    return('Can\'t understand model type')
                
                valuesInRange, response = model.checkValues(json)
                
                if not valuesInRange :
                    return jsonify({
                            'status' : 'error',
                            'prediction': response,
                            'is_json' : json
                            })
                else :
                    prediction = model.predictValueFromDict(json['formData'])[0][0]
                    return jsonify({
                                    'status' : 'success',
                                    'prediction': str(prediction),
                                    #'json' : str(json)
                                    })
            else :    
                return jsonify({
                            'status' : 'error',
                            'prediction': 'not a json',
                            'is_json' : request.is_json
                            })
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')
    
@application.route('/fields-buy-appartment', methods=['GET'])
def getFieldsWohnungKaufen():
        
    return jsonify(FIELDS_wohnung_kaufen)
    

@application.route('/fields-rent-appartment', methods=['GET'])
def getFieldsWohnungMieten():

    return jsonify(FIELDS_wohnung_mieten)
    
    
@application.route('/')
def test():
    
    return "Welcome to the real estate ai!"

    
def valueToType(value, value_type) :
    
    if value_type == 'float' :
        return float(value)
    elif value_type == 'int' :
        return int(value)
    else :
        return value
    
if __name__ == '__main__':
    
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    
    rentModel = RealEstatePredictor(MODELS_PATH_RENT, FIELDS_wohnung_mieten)
    buyModel = RealEstatePredictor(MODELS_PATH_BUY, FIELDS_wohnung_kaufen)
    print ('Model loaded')

    application.run(port=port, debug=True)
    
        
        
