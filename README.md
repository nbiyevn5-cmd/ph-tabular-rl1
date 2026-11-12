# Endüstriyel Atık Su Nötralizasyon Tanklarında Doğrusal Olmayan pH Dozalama Kontrolü için Tablo Tabanlı Pekiştirmeli Öğrenme Yaklaşımı

## Özet
Endüstriyel kimya ve tekstil tesislerinin atık su arıtma tesislerinde pH kontrolü, titrasyon eğrisinin şiddetli doğrusal olmayan (non-linear) karakteristiği ve hidrolik ölü zaman (time delay) gecikmeleri nedeniyle klasik PID veya Aç-Kapa denetleyiciler ile yönetilmesi zor bir süreçtir. Bu çalışmada, derin öğrenme (Deep Reinforcement Learning) mimarileri kullanılmaksızın, tamamen saf **Tabular Q-Learning (Pekiştirmeli Öğrenme)** algoritması ile adaptif bir pH doza ayarlama kontrolcüsü tasarlanmış ve simüle edilmiştir.

## 1. Giriş ve Endüstriyel Problem
Atık su nötralizasyon proseslerinde pH eğrisi logaritmik bir yapıya sahiptir; nötr bölge (pH 7 civarı) yakınlarında çok küçük bir reaktif ilavesi bile pH değerinde ani ve büyük sıçramalara (overshoot) yol açar. Ayrıca tank içi karıştırma gecikmeleri sistemde ölü zaman yaratır.

> **⚠️ Önemli Kriter Notu:** Projede **Derin Pekiştirmeli Öğrenme (Deep RL)** yöntemleri (LSTM tabanlı PPO, SAC vb.) kesinlikle **kullanılmamıştır**. Bunun yerine durum uzayı endüstriyel dinamiklere göre ayrıklaştırılarak klasik Tabular Q-Learning yöntemi tercih edilmiştir.

## 2. Markov Karar Süreci (MDP) Tasarımı
* **Durum (State):** Ayrıklaştırılmış pH sensör okuma kategorileri, eğilim (trend), pompa durumu ve ölü zamanı modellemek üzere geçmiş eylem yönü vektörü.
* **Eylem (Action):** Asit ve baz dozaj pompalarının kademeli debi değişim komutları.
* **Ödül Fonksiyonu (Reward):** Deşarj sınırları ($6.8 - 7.2$) içinde kalındığında pozitif sabit ödül ($+1$); sınır dışı sapmalarda karesel ceza ve kimyasal maliyeti.

## 3. Kurulum ve Çalıştırma
```bash
git clone [https://github.com/nbiyevn5-cmd/ph-tabular-rl1.git](https://github.com/nbiyevn5-cmd/ph-tabular-rl1.git)
cd ph-tabular-rl1
pip install numpy matplotlib
python ph_qlearning.py 40000
# ph-tabular-rl1