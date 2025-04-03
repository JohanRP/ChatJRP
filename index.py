from flask import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nombre_modelo="t5-large"
convertir_vectores=AutoTokenizer.from_pretrained(nombre_modelo)
modelo=AutoModelForSeq2SeqLM.from_pretrained(nombre_modelo)

app=Flask(__name__)

@app.route("/")
def root():
     return render_template('index.html')

@app.route('/resumir', methods=['GET','POST'])
def procesar():
     nombre = request.form['nombre']  # Extraer dato del input "nombre"
    
     texto=nombre

     texto="summarize: " + texto

     vectores_entrada=convertir_vectores.encode(texto,return_tensors="pt",max_length=1024,truncation=True)
     vectores_salida=modelo.generate(vectores_entrada,max_length=1024,min_length=10,length_penalty=2.0,num_beams=4, early_stopping=True)

     resumen=convertir_vectores.decode(vectores_salida[0],skip_special_tokens=True)
    
     return render_template('resumen.html', resumen=resumen)

@app.route('/resumen')
def resumen():
    return render_template('resumen.html')


if __name__=="__main__":
    app.run(debug=True)