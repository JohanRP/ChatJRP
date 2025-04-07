from flask import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

nombre_modelo="t5-large"
tipo_tarea=""

modeloP="valhalla/t5-base-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(modeloP)
model = T5ForConditionalGeneration.from_pretrained(modeloP)
 
convertir_vectores=T5Tokenizer.from_pretrained(nombre_modelo)
modelo=T5ForConditionalGeneration.from_pretrained(nombre_modelo)

app=Flask(__name__)

#----------Ruta Index----------
@app.route("/")
def root():
     return render_template('index.html')





#--------------------Apartado Traductor--------------------

     #----------Ruta Traducir----------
@app.route('/traductor')
def traductor():
     return render_template('traductor.html')

     #----------Funcion Traducir----------
@app.route('/traducir',methods=["GET","POST"])
def traducir():
     texto = request.form['texto']
     tipo_tarea="translate English to French: "
     texto=f"{tipo_tarea} {texto}"

     vectores_entrada=convertir_vectores(texto,return_tensors="pt").input_ids

     vectores_salida= modelo.generate(vectores_entrada,max_length=512)

     traduccion=convertir_vectores.decode(vectores_salida[0],skip_special_tokens=True)
     return render_template('traductor.html', traduccion=traduccion)


#--------------------Apartado Responder--------------------

     #----------Ruta Responder----------
@app.route('/responder')
def responder():
     return render_template('responder.html')

     #----------Funcion Respuesta----------
@app.route('/respuesta', methods=["GET", "POST"])
def respuesta():
     texto = request.form['texto']
     tipo_tarea="question: "
     texto=f"{tipo_tarea} {texto}"

     vectores_entrada=convertir_vectores(texto,return_tensors="pt").input_ids

     vectores_salida= modelo.generate(vectores_entrada,max_length=512)

     respuesta=convertir_vectores.decode(vectores_salida[0],skip_special_tokens=True)

     return render_template('responder.html',respuesta=respuesta)



#--------------------Apartado Generar--------------------

     #----------Ruta Generar----------
@app.route('/generar')
def generador():
     return render_template('generador.html')

     #----------Funcion Generar Pregunta----------
@app.route('/generarP', methods=["GET","POST"])
def generadorP():
     texto = request.form['texto']
     tipo_tarea="generate question: "
     texto=f"{tipo_tarea} {texto}"

     vectores_entrada=tokenizer(texto,return_tensors="pt").input_ids

     vectores_salida= model.generate(vectores_entrada,max_length=512)

     pregunta=tokenizer.decode(vectores_salida[0],skip_special_tokens=True)
     
     return render_template('generador.html', pregunta=pregunta)


#--------------------Apartado Resumen--------------------

     #----------Ruta Resumir----------
@app.route('/resumen')
def resumen():
    return render_template('resumen.html')

     #----------Funcion Resumir----------
@app.route('/resumir', methods=['GET','POST'])
def procesar():
     texto = request.form['nombre']  # Extraer dato del input "nombre"
    
     tipo_tarea="summarize: "
     texto=f"{tipo_tarea} {texto}"

     vectores_entrada=convertir_vectores(texto,return_tensors="pt").input_ids

     vectores_salida= modelo.generate(vectores_entrada,max_length=512)

     resumen=convertir_vectores.decode(vectores_salida[0],skip_special_tokens=True)

     return render_template('resumen.html', resumen=resumen)

#--------------------Apartado Imagen--------------------  

     #----------Ruta Resumir---------- 
@app.route('/imagen')
def imagen():
     return render_template('imagen.html')


if __name__=="__main__":
    app.run(debug=True)