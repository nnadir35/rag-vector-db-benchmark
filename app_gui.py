import logging

import gradio as gr
import httpx

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend Endpoint Ayarları
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/upload"
ASK_ENDPOINT = f"{API_URL}/ask"

async def upload_document(file_obj) -> str:
    """
    Kullanıcının seçtiği dosyayı FastAPI /upload endpoint'ine gönderir.
    Dönen chunks_added bilgisini veya olası hataları string olarak döndürür.
    """
    if file_obj is None:
        return "⚠️ Lütfen sisteme öğretmek için bir dosya seçin."

    try:
        async with httpx.AsyncClient() as client:
            # Dosyayı okuyup multipart/form-data olarak backend'e yolluyoruz
            with open(file_obj.name, "rb") as f:
                # Dosya adını güvenli bir şekilde alıp API'ye yüklüyoruz
                file_name = file_obj.name.split("/")[-1]
                files = {"file": (file_name, f)}

                # Embedding ve chunking uzun sürebileceği için timeout'u yüksek tutalım
                response = await client.post(UPLOAD_ENDPOINT, files=files, timeout=60.0)

            response.raise_for_status()
            data = response.json()
            chunks = data.get("chunks_added", 0)

            return f"✅ **Başarılı!** Sisteme başarıyla **{chunks}** parça (chunk) eklendi ve vektör veri tabanına işlendi."

    except httpx.ConnectError:
        logger.error("Upload - Backend bağlantı hatası.")
        return "❌ **Bağlantı Hatası:** Backend'e (localhost:8000) ulaşılamıyor. FastAPI sunucusunun açık olduğundan emin olun."
    except httpx.HTTPStatusError as e:
        logger.error(f"Upload - Server hatası: {e}")
        return f"❌ **Sunucu Hatası:** İşlem başarısız oldu (Statu Kodu: {e.response.status_code})."
    except Exception as e:
        logger.error(f"Upload - Beklenmeyen hata: {e}")
        return f"❌ **Beklenmeyen Hata:** {str(e)}"

async def ask_question(message: str, history: list[list[str]]) -> str:
    """
    Kullanıcının sorusunu FastAPI /ask endpoint'ine sorar.
    Gelen cevabın altına gecikme (latency) ve kaynak metin (retrieved context)
    bilgilerini estetik bir HTML formatında ekleyerek string döndürür.
    """
    if not message.strip():
        return "⚠️ Lütfen bir soru girin."

    try:
        async with httpx.AsyncClient() as client:
            payload = {"question": message}
            # GPT API çağrısı ve retriever süreci düşünülerek yüksek timeout
            response = await client.post(ASK_ENDPOINT, json=payload, timeout=60.0)
            response.raise_for_status()

            data = response.json()
            answer = data.get("answer", "Bir cevap üretilemedi.")
            latency = data.get("latency_seconds", 0)
            context_list = data.get("retrieved_context", [])

            # Kaynak metinler ve metrikler için şık, kapanıp açılabilen "details" bölümü
            metrics_html = f"<br><br><details><summary><small>⏱️ <b>Gecikme:</b> {latency:.2f} sn | 📚 <b>Kaynak Metin Parçaları ({len(context_list)})</b></small></summary>"
            metrics_html += "<small><ul style='margin-top: 10px; padding-left: 20px;'>"

            if not context_list:
                metrics_html += "<li>Bu cevap için herhangi bir kaynak kullanılmadı (Direct LLM Response).</li>"
            else:
                for idx, ctx in enumerate(context_list, 1):
                    content_str = str(ctx.get("content", ""))
                    safe_ctx = content_str.replace("<", "&lt;").replace(">", "&gt;")
                    metrics_html += f"<li style='margin-bottom: 8px;'><b>Kaynak {idx}:</b> {safe_ctx} (Skor: {ctx.get('score', 0):.4f})</li>"

            metrics_html += "</ul></small></details>"

            # Model asıl cevabını ve altında ekstra bilgileri döndür
            return answer + metrics_html

    except httpx.ConnectError:
        return "❌ **Bağlantı Hatası:** Backend'e ulaşılamıyor. Lütfen FastAPI sunucusunun ayakta olduğunu doğrulayın."
    except httpx.HTTPStatusError as e:
         return f"❌ **Sunucu Hatası:** Soru işlenirken {e.response.status_code} hatası alındı."
    except Exception as e:
         return f"❌ **Bilinmeyen Hata:** Cevap üretilirken bir hata oluştu: {str(e)}"

# ====== GRADIO ARAYÜZ TASARIMI ======
# "Soft" temasını kullanarak göze batmayan, temiz bir font ve modern bir UX elde ediyoruz
with gr.Blocks(title="AI RAG Asistanı") as demo:

    gr.Markdown(
        """
        # 🧠 Kurumsal RAG Asistanı (AI Engineer Edition)
        Eğitim materyallerinizi sisteme yükleyin, ardından sağ taraftaki sohbetten yapay zekaya dökümanlarla ilgili sorular sorun.
        """
    )

    with gr.Row():
        # Sol Kolon: Dosya Yükleme Paneli (Daha dar alan)
        with gr.Column(scale=3):
            gr.Markdown("### 📄 Sisteme Bilgi Öğret")
            gr.Markdown(
                "<small>PDF veya TXT uzantılı dosyalarınızı buradan yükleyerek vektör veri tabanını güncelleyebilirsiniz.</small>"
            )

            upload_component = gr.File(
                label="Dosya Sürükle veya Seç",
                file_types=[".pdf", ".txt"]
            )
            teach_btn = gr.Button("Sisteme Öğret 🚀", variant="primary")

            status_box = gr.Markdown("durum bilgisi bekleniyor...", visible=True)

            # Butona basıldığında upload fonksiyonunu tetikle
            teach_btn.click(
                fn=upload_document,
                inputs=[upload_component],
                outputs=[status_box]
            )

        # Sağ Kolon: Sohbet Arayüzü (Geniş alan)
        with gr.Column(scale=7):
            gr.Markdown("### 💬 RAG Chatbot")
            # gr.ChatInterface varsayılan olarak modern bir sohbet botu tasarımı sağlar.
            # fn parametresi ile asenkron fonksiyonumuza bağlıyoruz.
            chatbot = gr.ChatInterface(
                fn=ask_question,
                chatbot=gr.Chatbot(height=600),
                fill_height=False
            )

if __name__ == "__main__":
    # Konsola bilgi amaçlı yazdıralım
    print("🚀 Gradio UI başlatılıyor... (http://localhost:7860 adresini kontrol edin)")
    # Arayüzü tüm IP'lerden erişime açarak başlatıyoruz
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
