import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization, 
    Conv2DTranspose, Dropout, Activation, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# --- 1. Configuração e Constantes ---
BASE_PATH = '/home/clari/academics/salt_pepper2/images' 
NOISY_PATH = os.path.join(BASE_PATH, 'Noisy_folder')
CLEAN_PATH = os.path.join(BASE_PATH, 'Ground_truth')
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

# --- 2. Carregamento e Pré-processamento de Dados ---

def load_images_from_dir(noisy_dir, clean_dir):
    """
    Carrega pares de imagens (ruidosa, limpa) dos diretórios,
    redimensiona, normaliza E RETORNA OS CAMINHOS.
    """
    noisy_images = []
    clean_images = []
    noisy_paths = []
    clean_paths = []
    
    noisy_files = sorted(os.listdir(noisy_dir))
    
    print(f"Encontrados {len(noisy_files)} arquivos. Carregando pares...")
    
    for i, file_name in enumerate(noisy_files):
        noisy_img_path = os.path.join(noisy_dir, file_name)
        clean_file_name = file_name.replace("noisy_", "")
        clean_img_path = os.path.join(clean_dir, clean_file_name)
        
        noisy_img = cv2.imread(noisy_img_path, cv2.IMREAD_GRAYSCALE) 
        clean_img = cv2.imread(clean_img_path, cv2.IMREAD_GRAYSCALE)
        
        if noisy_img is None or clean_img is None:
            print(f"Aviso: Não foi possível carregar {file_name}. Pulando.")
            continue
            
        # Redimensiona para o TREINO
        noisy_img_resized = cv2.resize(noisy_img, (IMG_WIDTH, IMG_HEIGHT))
        clean_img_resized = cv2.resize(clean_img, (IMG_WIDTH, IMG_HEIGHT))
        
        noisy_images.append(noisy_img_resized)
        clean_images.append(clean_img_resized)
        
        # Guarda os caminhos originais
        noisy_paths.append(noisy_img_path)
        clean_paths.append(clean_img_path)
        
    print("Carregamento concluído.")
    
    # Converte as listas de imagens para arrays numpy
    noisy_images = np.array(noisy_images)
    clean_images = np.array(clean_images)
    
    noisy_images = noisy_images.astype('float32') / 255.0
    clean_images = clean_images.astype('float32') / 255.0

    noisy_images = np.expand_dims(noisy_images, axis=-1)
    clean_images = np.expand_dims(clean_images, axis=-1)
    
    # Retorna os arrays de imagem E as listas de caminhos
    return noisy_images, clean_images, noisy_paths, clean_paths

# Recebe os 4 outputs da função
X_noisy, Y_clean, ALL_noisy_paths, ALL_clean_paths = load_images_from_dir(NOISY_PATH, CLEAN_PATH)

print(f"Formato dos dados ruidosos (X): {X_noisy.shape}")
print(f"Formato dos dados limpos (Y): {Y_clean.shape}")
print(f"Total de caminhos carregados: {len(ALL_noisy_paths)}")

# Divide os dados em treino e teste (80% treino, 20% teste)
# Divide TUDO junto para manter a correspondência
X_train, X_test, Y_train, Y_test, noisy_paths_train, noisy_paths_test, clean_paths_train, clean_paths_test = train_test_split(
    X_noisy, Y_clean, ALL_noisy_paths, ALL_clean_paths, 
    test_size=0.2, 
    random_state=42
)

print(f"Imagens de Treino: {X_train.shape[0]}")
print(f"Imagens de Teste: {X_test.shape[0]}")
print(f"Caminhos de Teste: {len(noisy_paths_test)}")


# --- 3. Definição das Métricas (PSNR e SSIM) ---
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


# --- 4. Construção do Modelo (U-Net) ---
def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # --- Encoder (Codificador) ---
    c1 = Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)
    p1 = Dropout(0.3)(p1) 
    
    c2 = Conv2D(64, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    encoded = MaxPooling2D((2, 2), padding='same')(c2)
    
    # --- Decoder (Decodificador) ---
    u1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(encoded)
    u1 = concatenate([u1, c2]) 
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    u1 = Dropout(0.3)(u1)

    u2 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(u1)
    u2 = concatenate([u2, c1])
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', 
                     kernel_initializer='he_normal')(u2)

    model = Model(inputs, decoded)
    return model

autoencoder = build_unet((None, None, IMG_CHANNELS))

autoencoder.compile(
    optimizer='adam',
    loss='mean_absolute_error', 
    metrics=[psnr, ssim]
)

autoencoder.summary()


# --- 4.5. Pipeline de Dados com Augmentation ---

# Esta função aplica o mesmo augmentation para o par (ruidosa, limpa)
def augment(noisy_img, clean_img):
    # Empilha as duas imagens temporariamente
    stacked_images = tf.stack([noisy_img, clean_img], axis=0)
    
    # Aplica as transformações geométricas no stack
    # Isso garante que o flip seja o mesmo para ambas
    stacked_images = tf.image.random_flip_left_right(stacked_images)
    stacked_images = tf.image.random_flip_up_down(stacked_images)
    
    # Desempilha de volta
    noisy_img = stacked_images[0]
    clean_img = stacked_images[1]
    
    return noisy_img, clean_img

# Cria os objetos tf.data.Dataset
ds_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
ds_val = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# Define o tamanho do batch
BATCH_SIZE = 16 

# Cria o pipeline de TREINO
train_pipeline = ds_train.shuffle(buffer_size=X_train.shape[0]) \
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

# Cria o pipeline de VALIDAÇÃO (sem shuffle, sem augmentation)
val_pipeline = ds_val.batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


# --- 5. Treinamento do Modelo ---

# Callback
early_stopper = EarlyStopping(
    monitor='val_loss', 
    patience=10,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.0001
)

# Parâmetros de Treinamento
EPOCHS = 100

print("\nIniciando o treinamento com Data Augmentation...")

# --- Usamos os pipelines de dados no .fit() ---
history = autoencoder.fit(
    train_pipeline,
    epochs=EPOCHS,
    validation_data=val_pipeline,
    callbacks=[early_stopper],
    verbose=1
)

print("Treinamento concluído.")


# --- 6. Avaliação do Modelo ---

print("\nAvaliando o modelo no conjunto de teste...")
results = autoencoder.evaluate(val_pipeline, batch_size=BATCH_SIZE)

print(f"Resultados da Avaliação (Métricas):")
print(f"  Loss (MAE): {results[0]:.4f}")
print(f"  PSNR: {results[1]:.4f}")
print(f"  SSIM: {results[2]:.4f}")

# Salva o modelo
print("\nSalvando o modelo em 'denoiser_model.h5'...")
autoencoder.save('denoiser_model.h5')
print("Modelo salvo com sucesso!")


# --- 6.5. Função para Denoising de Imagens de Qualquer Tamanho ---

# Esta função pode ser usada na GUI para processar imagens maiores
def denoise_full_image(model, image_path):
    """
    Carrega uma imagem de qualquer tamanho, aplica o modelo de denoise
    e retorna a imagem limpa no tamanho original.
    """
    # 1. Carrega a imagem original
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar {image_path}")
        return None
        
    original_h, original_w = img.shape
    
    # 2. Normaliza
    img_normalized = img.astype('float32') / 255.0
    
    # 3. Calcula o padding necessário (para ser múltiplo de 4)
    # (A  U-Net tem 2 níveis de pooling, 2*2=4)
    PAD_FACTOR = 4 
    h_pad = (PAD_FACTOR - original_h % PAD_FACTOR) % PAD_FACTOR
    w_pad = (PAD_FACTOR - original_w % PAD_FACTOR) % PAD_FACTOR
    
    # Adiciona padding (usando reflexão para bordas melhores)
    padded_img = cv2.copyMakeBorder(
        img_normalized, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT
    )
    
    # 4. Prepara para o modelo (adiciona dimensão de batch e canal)
    input_tensor = np.expand_dims(padded_img, axis=-1)  # (H, W, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0) # (1, H, W, 1)
    
    # 5. Executa a predição
    pred_padded = model.predict(input_tensor)
    
    # 6. Remove o padding da imagem de saída
    pred_original_size = pred_padded[0, 0:original_h, 0:original_w, 0]
    
    # 7. Desnormaliza (volta para 0-255)
    img_denoised = (pred_original_size * 255.0).astype(np.uint8)
    
    return img_denoised

# --- 7. Visualização dos Resultados ---

# Salva o modelo (como no seu script original)
print("\nSalvando o modelo em 'denoiser_model.h5'...")
autoencoder.save('denoiser_model.h5')
print("Modelo salvo com sucesso!")

# --- Visualização 1: Predição em Tamanho Real ---
print("\nGerando predições em TAMANHO REAL para visualização...")

n = 5 # Número de imagens para mostrar
if len(noisy_paths_test) < n:
    print(f"Aviso: Menos de {n} imagens de teste. Mostrando {len(noisy_paths_test)}.")
    n = len(noisy_paths_test)

plt.figure(figsize=(20, 12))
plt.suptitle("Comparação (Tamanho Real vs Predição)", fontsize=16)

for i in range(n):
    # Pega os caminhos
    noisy_path = noisy_paths_test[i]
    clean_path = clean_paths_test[i]
    
    # Carrega a imagem ruidosa original (para display)
    noisy_img_original = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    
    # Usando a função de denoise para tamanho real
    predicted_img = denoise_full_image(autoencoder, noisy_path)
    
    # Carrega a imagem limpa original (para display)
    clean_img_original = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

    if noisy_img_original is None or predicted_img is None or clean_img_original is None:
        print(f"Erro ao carregar/processar imagem {noisy_path}, pulando visualização.")
        continue

    # Imagem Ruidosa (Entrada Original)
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(noisy_img_original, cmap='gray')
    plt.title(f"Ruidosa (Original)\n{noisy_img_original.shape}")
    plt.axis('off')

    # Imagem Corrigida (Predição)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(predicted_img, cmap='gray')
    plt.title(f"Corrigida (Predição)\n{predicted_img.shape}")
    plt.axis('off')

    # Imagem Real (Ground Truth Original)
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(clean_img_original, cmap='gray')
    plt.title(f"Real (Original)\n{clean_img_original.shape}")
    plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results_comparison_fullsize.png') 
plt.close()
print(f"Gráfico 'results_comparison_fullsize.png' salvo!")


# --- Visualização 2: Histórico de Treinamento ---

print("Gerando gráficos do histórico de treinamento...")
plt.figure(figsize=(18, 5))
plt.suptitle("Histórico de Treinamento", fontsize=16)

# Gráfico da Loss (Perda)
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.title('Histórico da Perda (MAE)')
plt.xlabel('Época')
plt.ylabel('Loss (MAE)')
plt.legend()

# Gráfico do PSNR
plt.subplot(1, 3, 2)
plt.plot(history.history['psnr'], label='PSNR Treino')
plt.plot(history.history['val_psnr'], label='PSNR Validação')
plt.title('Histórico do PSNR')
plt.xlabel('Época')
plt.ylabel('PSNR (dB)')
plt.legend()

# Gráfico do SSIM
plt.subplot(1, 3, 3)
plt.plot(history.history['ssim'], label='SSIM Treino')
plt.plot(history.history['val_ssim'], label='SSIM Validação')
plt.title('Histórico do SSIM')
plt.xlabel('Época')
plt.ylabel('SSIM')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig('training_history.png')
plt.close()

print("\nGráficos de resultado e histórico salvos!")
print("Script finalizado com sucesso.")