let modelo;
let vocab = [];
let vocabInverso = {};
const MAX_LEN = 512; 

const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const statusSpan = document.getElementById('status');

// 1. Inicializar carregando os arquivos como GRAPH MODEL
async function inicializar() {
    try {
        const respVocab = await fetch('modelo/vocab.json');
        vocab = await respVocab.json();
        
        vocab.forEach((palavra, index) => {
            vocabInverso[index] = palavra;
        });

        // Correção definitiva aplicada aqui: loadGraphModel
        modelo = await tf.loadGraphModel('modelo/model.json');

        statusSpan.innerText = "🟢 Pronto";
        userInput.disabled = false;
        sendBtn.disabled = false;
    } catch (erro) {
        console.error(erro);
        statusSpan.innerText = "❌ Erro ao carregar";
    }
}

function tokenizar(texto) {
    const textoLimpo = texto.toLowerCase().replace(/([?.!,])/g, " $1 ");
    const palavras = textoLimpo.trim().split(/\s+/);
    
    let ids = [vocab.indexOf('<bos>'), vocab.indexOf('<user>'), ...palavras.map(p => {
        const id = vocab.indexOf(p);
        return id !== -1 ? id : vocab.indexOf('[UNK]');
    }), vocab.indexOf('<bot>')].filter(id => id !== -1);

    while (ids.length < MAX_LEN) {
        ids.push(0);
    }
    return ids.slice(0, MAX_LEN);
}

// 3. Inferência adaptada para modelos de Grafo (.execute)
async function gerarResposta(prompt) {
    let inputIds = tokenizar(prompt);
    let tokensGerados = [];
    const maxTokensNovos = 50;
    const temperature = 0.7;

    for (let i = 0; i < maxTokensNovos; i++) {
        const inputTensor = tf.tensor2d([inputIds], [1, MAX_LEN], 'int32');
        
        // Modelos de Grafo usam obrigatoriamente .execute()
        const predicoes = modelo.execute(inputTensor);
        
        const dadosPredicao = await predicoes.data();
        const numClasses = 32000; 
        const idxUltimoToken = (MAX_LEN - 1) * numClasses;
        let logitsUltimo = Array.from(dadosPredicao.slice(idxUltimoToken, idxUltimoToken + numClasses));

        logitsUltimo = logitsUltimo.map(v => Math.log(v + 1e-8) / temperature);
        const expLogits = logitsUltimo.map(Math.exp);
        const somaExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(v => v / somaExp);

        let idPrevisto = 0;
        let r = Math.random();
        let somaAcumulada = 0;
        for (let j = 0; j < probs.length; j++) {
            somaAcumulada += probs[j];
            if (r <= somaAcumulada) {
                idPrevisto = j;
                break;
            }
        }

        inputTensor.dispose();
        predicoes.dispose();

        if (idPrevisto === 0 || idPrevisto === vocab.indexOf('<eos>')) {
            break;
        }

        tokensGerados.push(idPrevisto);

        inputIds.shift();
        inputIds.push(idPrevisto);
    }

    const respostaTexto = tokensGerados.map(id => vocabInverso[id] || "").join(" ");
    return respostaTexto.trim();
}

function appendMessage(texto, remetente) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${remetente}-message`);
    msgDiv.innerText = texto;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function lidarComEnvio() {
    const texto = userInput.value.trim();
    if (!texto) return;

    appendMessage(texto, 'user');
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    statusSpan.innerText = "⚡ Pensando...";

    const resposta = await gerarResposta(texto);
    
    appendMessage(resposta || "... (não entendi)", 'bot');
    userInput.disabled = false;
    sendBtn.disabled = false;
    statusSpan.innerText = "🟢 Pronto";
    userInput.focus();
}

sendBtn.addEventListener('click', lidarComEnvio);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') lidarComEnvio(); });

inicializar();
