from django.shortcuts import render
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Cargar el modelo y el tokenizador
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

# Función para la corrección gramatical
def corregir_gramatica(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return edited_text

# Vista para la página principal
def index(request):
    resultado = None
    texto_original = None
    if request.method == 'POST':
        input_text = request.POST.get('texto', '')
        texto_original = input_text
        edited_text = corregir_gramatica(input_text)
        resultado = edited_text
    return render(request, 'index.html', {'resultado': resultado, 'texto_original': texto_original})
