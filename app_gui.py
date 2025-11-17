import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

# --- 1. Configuração ---
MODEL_PATH = 'denoiser_model.h5'
PAD_FACTOR = 4 # O modelo tem 2 níveis de pooling (2*2 = 4)

# --- 2. Funções de Métrica (Necessárias para Carregar o Modelo) ---
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# --- 3. Carregar o Modelo Treinado ---
if not os.path.exists(MODEL_PATH):
    print(f"Erro: Arquivo do modelo '{MODEL_PATH}' não encontrado.")
    print("Por favor, execute o script de treino primeiro para gerar este arquivo.")
    exit()

try:
    # Informa ao Keras sobre as funções customizadas
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'psnr': psnr, 'ssim': ssim}
    )
    print(f"Modelo '{MODEL_PATH}' carregado com sucesso.")
    model.summary() # Bom para verificar se a input shape é (None, None, None, 1)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# --- 4. Função Principal da GUI (Predição em Tamanho Real) ---

def denoise_image_full_size(input_image):
    """
    Recebe uma imagem (numpy array), processa em tamanho real
    e retorna a imagem limpa (numpy array).
    """
    
    # 1. Pré-processamento: Garante escala de cinza
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    else:
        # Assume que já está em escala de cinza
        img_gray = input_image
    
    # Guarda o tamanho original para o crop final
    original_h, original_w = img_gray.shape
    
    # 2. Normaliza (0-255 -> 0.0-1.0)
    img_normalized = img_gray.astype('float32') / 255.0
    
    # 3. Calcula o padding necessário (lógica do seu script)
    h_pad = (PAD_FACTOR - original_h % PAD_FACTOR) % PAD_FACTOR
    w_pad = (PAD_FACTOR - original_w % PAD_FACTOR) % PAD_FACTOR
    
    # 4. Adiciona padding (usando reflexão para bordas melhores)
    padded_img = cv2.copyMakeBorder(
        img_normalized, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT
    )
    
    # 5. Prepara para o modelo (adiciona dimensão de batch e canal)
    input_tensor = np.expand_dims(padded_img, axis=-1)  # (H_pad, W_pad, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0) # (1, H_pad, W_pad, 1)
    
    # 6. Executa a predição
    pred_padded = model.predict(input_tensor)
    
    # 7. Remove o padding da imagem de saída (lógica do seu script)
    # pred_padded tem shape (1, H_pad, W_pad, 1)
    pred_original_size = pred_padded[0, 0:original_h, 0:original_w, 0]
    
    # 8. Desnormaliza (volta para 0-255)
    img_denoised = np.clip(pred_original_size * 255.0, 0, 255).astype(np.uint8)
    
    return img_denoised

# --- 5. Criar e Lançar a Interface Gradio ---
print("Lançando a interface Gradio...")

iface = gr.Interface(
    fn=denoise_image_full_size,
    inputs=gr.Image(label="Upload Imagem Ruidosa", type="numpy"),
    outputs=gr.Image(label="Imagem Corrigida (Denoised)", type="numpy"),
    title="Removedor de Ruído de Imagens (CNN U-Net)",
    description='''
    <p style="text-align: center;">
        Trabalho realizado para nota na matéria "Tópicos Avançados em Computação".
    </p>
    <p style="text-align: center;">
        Grupo: Clariana Costa e Lara Marques.
    </p>
    <br>
    <p>
        Faça o upload de uma imagem com ruído. O modelo processará a imagem em sua resolução original, sem redimensionar.
    </p>
    '''
)

iface.launch()