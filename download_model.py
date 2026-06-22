import urllib.request
import os
import time
import sys

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
ALT_MODEL_URL = "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/testdata/hand_landmarker.task"

def download_with_progress(url, dest, max_retries=5):
    """Download file dengan progress bar dan retry otomatis"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n🔄 Percobaan {attempt}/{max_retries}")
            print(f"📡 URL: {url}")
            
            # Request dengan User-Agent agar tidak diblokir
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Buka koneksi dengan timeout
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                block_size = 8192  # 8KB per chunk
                
                print(f"📦 Ukuran file: {total_size / (1024*1024):.2f} MB")
                print("📥 Downloading...\n")
                
                # Download ke file temporary dulu
                temp_dest = dest + '.tmp'
                with open(temp_dest, 'wb') as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Tampilkan progress bar
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            mb_down = downloaded / (1024*1024)
                            mb_total = total_size / (1024*1024)
                            
                            # Progress bar visual
                            bar_length = 40
                            filled = int(bar_length * downloaded / total_size)
                            bar = '█' * filled + '░' * (bar_length - filled)
                            
                            print(f"\r[{bar}] {percent:5.1f}% | {mb_down:.1f}/{mb_total:.1f} MB", 
                                  end='', flush=True)
                
                # Rename file temporary ke nama final
                os.replace(temp_dest, dest)
                print(f"\n\n✅ Download berhasil!")
                print(f"📁 File tersimpan: {os.path.abspath(dest)}")
                print(f"📏 Ukuran: {os.path.getsize(dest) / (1024*1024):.2f} MB")
                return True
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Download dibatalkan oleh user (Ctrl+C)")
            if os.path.exists(dest + '.tmp'):
                os.remove(dest + '.tmp')
            return False
            
        except Exception as e:
            print(f"\n\n❌ Gagal: {e}")
            if os.path.exists(dest + '.tmp'):
                os.remove(dest + '.tmp')
            
            if attempt < max_retries:
                # Coba URL alternatif jika URL utama gagal
                if url == MODEL_URL:
                    print("🔄 Beralih ke URL alternatif...")
                    url = ALT_MODEL_URL
                else:
                    wait_time = attempt * 3
                    print(f"⏳ Menunggu {wait_time} detik sebelum mencoba lagi...")
                    time.sleep(wait_time)
    
    return False

def main():
    print("=" * 60)
    print("📥 DOWNLOAD MODEL MEDIAPIPE HAND LANDMARKER")
    print("=" * 60)
    
    # Cek apakah file sudah ada
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
        print(f"\n✅ File '{MODEL_PATH}' sudah ada!")
        print(f"📏 Ukuran: {size_mb:.2f} MB")
        
        response = input("\nApakah ingin download ulang? (y/n): ").strip().lower()
        if response != 'y':
            print("👍 Tidak jadi download ulang.")
            return
    
    # Download file
    success = download_with_progress(MODEL_URL, MODEL_PATH)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SELESAI! Kamu bisa menjalankan program utama:")
        print("   uv run foto_kita_blur.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Download gagal setelah beberapa percobaan.")
        print("\n💡 Solusi alternatif:")
        print("   1. Download manual via browser:")
        print(f"      {MODEL_URL}")
        print(f"      atau {ALT_MODEL_URL}")
        print("   2. Gunakan VPN jika URL diblokir")
        print("   3. Minta teman download dan kirim filenya")
        print(f"   4. Letakkan file '{MODEL_PATH}' di folder ini")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()