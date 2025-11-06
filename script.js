// script.js
let model = null;
const MODEL_PATH = "model/model.json"; // asegúrate de que esta ruta corresponda
const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const predList = document.getElementById("pred-list");
const autoPredict = document.getElementById("auto-predict");

// Inicializar canvas negro con stroke blanco ancho
function initCanvas(){
  ctx.fillStyle = "black";
  ctx.fillRect(0,0,canvas.width, canvas.height);
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
}
initCanvas();

// Dibujo táctil/mouse
let drawing = false;
function getPos(e){
  const rect = canvas.getBoundingClientRect();
  if (e.touches) {
    const t = e.touches[0];
    return {x: t.clientX - rect.left, y: t.clientY - rect.top};
  } else {
    return {x: e.clientX - rect.left, y: e.clientY - rect.top};
  }
}
canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
});
canvas.addEventListener("pointerup", (e) => {
  drawing = false;
  if (autoPredict.checked) predict();
});
canvas.addEventListener("pointerleave", () => { drawing = false; });

// Limpiar canvas
clearBtn.addEventListener("click", () => {
  initCanvas();
  showMessage("Canvas limpio");
});

// Cargar modelo TFJS
async function loadModel(){
  predList.innerHTML = "Cargando modelo...";
  try {
    model = await tf.loadLayersModel(MODEL_PATH);
    // Si tu modelo espera entrada [1,28,28,1] con valores entre 0-1, está listo.
    predList.innerHTML = "Modelo cargado. Dibuja un número y presiona Predecir.";
    console.log("Modelo cargado:", model);
  } catch(err){
    predList.innerHTML = "Error cargando modelo. Revisa consola y ruta.";
    console.error(err);
  }
}
loadModel();

// Preprocesamiento: canvas -> tensor (1,28,28,1)
function preprocessCanvas(image){
  // image: HTMLCanvasElement
  return tf.tidy(() => {
    // tomamos la imagen, la convertimos a tensor RGBA
    let tensor = tf.browser.fromPixels(image, 1); // 1 canal (grayscale)
    // canvas es 280x280; resize a 28x28
    tensor = tf.image.resizeBilinear(tensor, [28,28]);
    // tensor shape [28,28,1] valores 0-255 -> normalizamos 0-1
    tensor = tensor.toFloat().div(255.0);
    // Invertir colores: en MNIST el fondo es negro y el dígito blanco. 
    // Si tu canvas tiene fondo negro y trazo blanco, no necesitas invertir.
    // A veces conviene invertir dependiendo del modelo; aquí asumimos: blanco sobre negro -> OK.
    // Si tu modelo fue entrenado con blanco fondo y negro trazos, usar: tensor = tf.sub(1.0, tensor);
    // Añadir batch dim:
    return tensor.expandDims(0); // [1,28,28,1]
  });
}

// Mostrar predicciones (top 3)
function showPreds(predsArray){
  // predsArray es un array de 10 probabilidades
  const pairs = predsArray.map((p,i)=>({digit:i, prob:p}));
  pairs.sort((a,b)=>b.prob - a.prob);
  predList.innerHTML = "";
  for(let i=0;i<3;i++){
    const p = pairs[i];
    const div = document.createElement("div");
    div.className = "pred-item";
    div.innerHTML = `<div class="pred-digit">${p.digit}</div><div class="pred-prob">${(p.prob*100).toFixed(1)}%</div>`;
    predList.appendChild(div);
  }
}

// Mensajes simples
function showMessage(txt){
  predList.innerHTML = `<div class="pred-item"><div>${txt}</div></div>`;
}

// Predict button
predictBtn.addEventListener("click", predict);

async function predict(){
  if (!model) { showMessage("Modelo no cargado"); return; }
  showMessage("Procesando...");
  await tf.nextFrame();
  // Crear un canvas temporal con el contenido (asegurar tamaño)
  // Preprocesar
  const tensor = preprocessCanvas(canvas); // [1,28,28,1]
  const preds = model.predict(tensor);
  // obtener array
  const data = await preds.data();
  showPreds(Array.from(data));
  tf.dispose([tensor, preds]);
}
