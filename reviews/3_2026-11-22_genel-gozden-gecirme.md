# 3 · Site ve Python panelinin genel gözden geçirmesi

- **Tarih:** 2026-11-22
- **Commit:** `6898057` (Release 1.0.1: fixes from a general review of the site and dashboard)
- **Not:** Bu kayıt 2026-11-26'da, o günkü rapordan ve [CHANGELOG.md](../CHANGELOG.md)'deki 1.0.1
  bölümünden sonradan yazıldı. Ayrıntılı liste CHANGELOG'da.

## Kapsam

Bütün site (üç bölüm, tarayıcıda CNN eğitimi, gezinme, erişilebilirlik) ve Python paneli
(`dashboard.py`, `start.bat`, kurulum).

## Bulgular

| No | Bulgu | Durum |
|---|---|---|
| G1 | Güvenlik: açık bir web sitesi, panel çalışırken eğitimi başlatıp modelin üzerine yazabiliyordu | Çözüldü (1.0.1) |
| G2 | Tarayıcıda CNN eğitimi: her batch'te bellek sızıyordu; GPU kilitlenince sayfa yenilenene kadar eğitim çalışmıyordu | Çözüldü (1.0.1) |
| G3 | CNN Teori: hızlı model değişiminde yanlış ağırlıklar kalabiliyordu; uç değerlerde kayıp yanlış çıkıyordu | Çözüldü (1.0.1) |
| G4 | Perceptron: "Yeniden üret" elle konan noktaları siliyordu; hesapta yanlış öğrenme oranı görünebiliyordu | Çözüldü (1.0.1) |
| G5 | MLP: kayıp demosu minimuma varmadan "minimumda ✓" diyordu | Çözüldü (1.0.1) |
| G6 | Gezinme: Geri tuşu yerini kaybediyordu; yenileyince okuma konumu gidiyordu; bozuk adres sayfayı bozuyordu | Çözüldü (1.0.1) |
| G7 | Erişilebilirlik: düşük kontrast, kaybolan klavye odağı | Çözüldü (1.0.1) |
| G8 | Kurulum: `start.bat` yarım kurulumda takılıyordu; macOS/Linux adımları yoktu | Çözüldü (1.0.1) |
| G9 | Eğitim günlüğü iki dilde de İngilizce | Bırakıldı: günlük, PyTorch betiğinin kendi çıktısıyla aynı kalsın |
| G10 | Mac'te PyTorch eğitimi CPU'da sürüyor (MPS kullanılmıyor) | Bırakıldı: MPS desteği, Kodlama sekmesinde gösterilen `SimpleCNN.py` satırlarını değiştirirdi |
| G11 | Çok uzun eğitimlerde panelin durum yanıtı büyüyor | Ertelendi: varsayılan ayarlarda sorun değil |

## Doğrulanamayanlar ve sonraki durumu

| Konu | Durum |
|---|---|
| Firefox ve Safari | Çözüldü (1.0.2): Firefox, WebKit ve öykünülmüş Android/iPhone testleri eklendi. Gerçek iPhone ve Android cihazda deneme hâlâ yapılmadı |
| GitHub'daki Python işi | Çözüldü (2026-11-26): iş akışının main'e gelen push'taki çalışmasında başarılı |
| Gerçek ekran okuyucuyla deneme | **Açık** |
| Gizli sekmede eğitimin sürmesi, `start.bat`'in yeni kurulum yolu | **Açık:** otomasyonla denenemedi, elle bir kez denenmeli |
