let modelo;
let vocab = [];
let vocabInverso = {};
const MAX_LEN = 512; // Alinhar com o tamanho do seu modelo

// Elementos da Interface
const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const statusSpan = document.getElementById('status');

// 1. Inicializar o Site carregando os arquivos
async function inicializar() {
    try {
        // Carrega o Vocabulário
        const respVocab = await fetch('modelo/vocab.json');
        vocab = await respVocab.json();
        
        // Cria o mapeamento inverso (ID -> Palavra)
        vocab.forEach((palavra, index) => {
            vocabInverso[index] = palavra;
        });

        // Carrega o Modelo do TensorFlow.js
        modelo = await tf.loadGraphModel('modelo/model.json');

        // Ativa os campos de texto
        statusSpan.innerText = "🟢 Pronto";
        userInput.disabled = false;
        sendBtn.disabled = false;
    } catch (erro) {
        console.error(erro);
        statusSpan.innerText = "❌ Erro ao carregar";
    }
}

// 2. Tokenizer Manual Simplificado (Limpa e converte texto em IDs)
function tokenizar(texto) {
    // Padronização básica: caixa baixa e separa pontuações por espaço
    const textoLimpo = texto.toLowerCase().replace(/([?.!,])/g, " $1 ");
    const palavras = textoLimpo.trim().split(/\s+/);
    
    let ids = [vocab.indexOf('<bos>'), vocab.indexOf('<user>'), ...palavras.map(p => {
        const id = vocab.indexOf(p);
        return id !== -1 ? id : vocab.indexOf('[UNK]');
    }), vocab.indexOf('<bot>')].filter(id => id !== -1);

    // Preenche com PAD (0) até chegar no MAX_LEN
    while (ids.length < MAX_LEN) {
        ids.push(0);
    }
    // Corta se passar do tamanho máximo
    return ids.slice(0, MAX_LEN);
}

// 3. Função de Inferência (Gera o texto da resposta)
async function gerarResposta(prompt) {
    let inputIds = tokenizar(prompt);
    let tokensGerados = [];
    const maxTokensNovos = 50;
    const temperature = 0.7;

    for (let i = 0; i < maxTokensNovos; i++) {
        // Converte o array para um Tensor do TensorFlow
        const inputTensor = tf.tensor2d([inputIds], [1, MAX_LEN], 'int32');
        
        // Roda a previsão
        const predicoes = modelo.execute(inputTensor);
        
        // Pega as probabilidades do ÚLTIMO token da sequência
        const dadosPredicao = await predicoes.data();
        const numClasses = 32000; // Tamanho do vocabulário
        const idxUltimoToken = (MAX_LEN - 1) * numClasses;
        let logitsUltimo = Array.from(dadosPredicao.slice(idxUltimoToken, idxUltimoToken + numClasses));

        // Aplica a Temperatura
        logitsUltimo = logitsUltimo.map(v => Math.log(v + 1e-8) / temperature);
        const expLogits = logitsUltimo.map(Math.exp);
        const somaExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(v => v / somaExp);

        // Amostragem (Categorical Sampling simples)
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

        // Limpa tensores da memória para não travar o navegador
        inputTensor.dispose();
        predicoes.dispose();

        // Se prever o fim da frase (<eos>) ou PAD (0), para a geração
        if (idPrevisto === 0 || idPrevisto === vocab.indexOf('<eos>')) {
            break;
        }

        tokensGerados.push(idPrevisto);

        // Desloca a janela do input para a direita para prever o próximo token
        inputIds.shift();
        inputIds.push(idPrevisto);
    }

    // Converte os IDs gerados de volta para palavras
    const respostaTexto = tokensGerados.map(id => vocabInverso[id] || "").join(" ");
    return respostaTexto.trim();
}

// 4. Controle de Mensagens na Tela
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

// Eventos de clique e teclado
sendBtn.addEventListener('click', lidarComEnvio);
userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') lidarComEnvio(); });

// Inicia o processo ao abrir a página
inicializar();
