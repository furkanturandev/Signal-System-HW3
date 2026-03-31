# Speech Gender Classifier

Bu proje, zaman-domeni oto-korelasyon (autocorrelation) yöntemi kullanarak konuşma sinyallerinden temel frekans (F0) çıkarımı yapar ve basit bir kural tabanlı algoritma ile konuşmacının cinsiyet sınıfını (Male / Woman / Child) tahmin eder.

Dosya
- `speech_gender_classifier.py` — Tek dosyalık Streamlit uygulaması. Ana analiz ve arayüz burada.

Gereksinimler
- Python 3.8+ (tercihen 3.9 veya 3.10)
- Aşağıdaki paketler gereklidir; bir `requirements.txt` dosyası sağlanmıştır.

Kurulum (Windows PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Çalıştırma

Streamlit arayüzünü çalıştırmak için:

```powershell
streamlit run speech_gender_classifier.py
```

Kullanım Özeti
- Sağ kenar paneline (sidebar) veri seti klasörünün tam yolunu girin. Bu klasörün içinde `Group_01`, `Group_02`, ... veya benzeri alt klasörler olmalıdır.
- Kod, varsayılan olarak şu dosya şablonlarını arar (sırasıyla):
  1. `**/Grup_*.xlsx`
  2. `**/Group_*.xlsx`
  3. `**/*.xlsx` (eğer yukarıdakiler yoksa tüm Excel dosyaları)
- Excel dosyalarındaki kolon isimleri normalize edilir. Önemli kolonlar:
  - `File_Name` veya benzeri (dosya adı ya da yol)
  - `Gender` (cinsiyet) — değerler "Male", "Woman" ya da "Child" ya da Türkçe karşılıkları olabilir
- Uygulama hem tek dosya yükleyerek anlık tahmin yapmaya, hem de Excel metadata tabanlı tüm veri kümesi analizi yapmaya uygundur.

Excel ve .wav Dosyaları Hakkında Notlar
- Excel içindeki `File_Name` sütunundaki değerler ya tam yol ya da yalnızca dosya adını içerebilir. Kod aşağıdaki yolları kontrol eder:
  - Doğrudan belirtilen yol
  - Dataset root ile birleştirilmiş yol
  - Excel dosyasının bulunduğu klasör
  - Tüm alt klasörlerde dosya adı araması
- Eğer .wav dosyaları bulunamazsa, analiz atlanır ve ilk birkaç atlanan örnek kullanıcıya gösterilir.

Sorun Giderme
- "No metadata Excel files found" hatası alıyorsanız, dataset klasör yolunu doğru girdiğinizden ve Excel dosyalarının isimlerinin `Grup_` veya `Group_` ile başladığından emin olun.
- Sessiz / çok kısa .wav dosyalar analiz edilemeyebilir.
- Eğer `openpyxl` veya `librosa` ile ilgili hata alırsanız, paketlerin doğru kurulduğunu kontrol edin.

Geliştirme Notları
- Temel F0 tahmini oto-korelasyon yöntemi ile yapılıyor; karşılaştırma için FFT tabanlı bir yöntem de hesaplanıp raporda gösteriliyor.
- Kural tabanlı sınıflandırma için eşikler `speech_gender_classifier.py` içinde tanımlıdır (MALE_UPPER, WOMAN_UPPER). Sidebar üzerinden kullanıcı tarafından değiştirilebilir.

Lisans
- Proje için özel bir lisans belirtilmemiştir. Kendi çalışmanız için uygun lisansı ekleyin.

İletişim
- Daha fazla değişiklik veya ilave isterseniz, repository içinde değişiklik yapabilirim.