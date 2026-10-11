---
template: home.html
title: yonga — Hugging Face modellerini Intel AI PC'lerde çalıştırın
description: >-
  yonga makinenizi inceler, NPU, GPU veya CPU için doğrulanmış bir OpenVINO modeli hazırlar
  ve bir şey çalışamadığında nedenini söyler.
hide:
  - navigation
  - toc
home:
  badge: v0.5.1 · ilk herkese açık sürüm
  badge_link: https://github.com/ooguz/yonga/releases/tag/v0.5.1
  title: Hugging Face modelleri, Intel <em>NPU, GPU ve CPU</em>'nuzda.
  lead: >-
    yonga'ya bir model kimliği verin. Makinenizi inceler, bir aygıt ve nicemleme profili seçer,
    modeli OpenVINO ile dönüştürür, gerçekten çalıştığını sınar ve derleyici hatası yığını
    dökmek yerine her geri dönüşü açıklar.
  primary: Başlayın
  primary_link: getting-started/
  secondary: GitHub'da görün
  install_label: Kurulum
  install_tabs:
    - name: AppImage
      code: |-
        curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
        chmod +x yonga-0.5.1-x86_64.AppImage
        ./yonga-0.5.1-x86_64.AppImage self install
    - name: pip
      code: |-
        python3 -m venv ~/.venvs/yonga
        ~/.venvs/yonga/bin/pip install "yonga[openvino,export,serve]"
  install_note: Linux x86_64 · sudo gerekmez · ev dizininizin dışına hiçbir şey yazılmaz
  terminal_caption: Intel Core Ultra 7 268V üzerinde gerçek bir çalıştırma, kısaltılmış.
  steps_title: Model kimliğinden yanıta üç komut
  steps:
    - command: yonga inspect Qwen/Qwen3-0.6B
      text: Daha hiçbir şey indirilmeden mimariyi, boyutu, lisansı ve modeli bu makinede hazırlamanın neler gerektireceğini görün.
    - command: yonga prepare Qwen/Qwen3-0.6B
      text: Yalnızca dönüştürmenin ihtiyaç duyduğu dosyaları indirir, nicemler, bilinen sorunları onarır ve hedef aygıtta bir duman testi derler.
    - command: yonga chat Qwen/Qwen3-0.6B
      text: Terminalde sohbet edin ya da mevcut araçlarınızın konuşabileceği OpenAI uyumlu bir API için <code>yonga serve</code> çalıştırın.
  features_title: Zaten sahip olduğunuz donanım için
  features:
    - icon: material/chip
      title: Aygıtı tanır, NPU dahil
      text: yonga NPU'yu, GPU'yu ve CPU'yu yoklar, aygıtın kaldırdığı bilinen bir profili seçer ve siz bir aygıt belirttiğinizde arkanızdan başka aygıta geçmez.
    - icon: material/check-decagram-outline
      title: Doğrulanmış, yeniden üretilebilir modeller
      text: Her artifact kaynak revizyonunu sabitler, profili ve dayanağını kaydeder, dosya özetlerini taşır ve hazır sayılmadan önce gerçek bir üretim testinden geçer.
    - icon: material/message-alert-outline
      title: Üzerinde işlem yapılabilen hatalar
      text: Veriye dayalı bir kayıt defteri derleyici, dışa aktarıcı ve sürücü hatalarını tanır; bunları kodu, nedeni ve çalıştırılacak sonraki komutu olan bir tanılamaya dönüştürür.
    - icon: material/api
      title: OpenAI uyumlu sunucu
      text: Akış (streaming), durdurma dizileri, desteklenen ailelerde araç çağırma ve görü modelleri için görseller; hepsi tek bir yerel artifact'ten sunulur.
    - icon: material/shield-lock-outline
      title: Tasarımı gereği gizli
      text: Telemetri yok. yonga OpenVINO'nun kendi kullanım raporlamasını da engeller, Hugging Face Hub'a yalnızca gerektiğinde bağlanır ve önbellekte olanla çevrimdışı çalışabilir.
    - icon: material/package-variant-closed
      title: Kurulacak tek dosya
      text: AppImage, Python'u ve OpenVINO'yu içinde taşır. Dışa aktarmalar yönetilen ortamlarda çalışır; sistem Python'unuza dokunulmaz, root yetkisi gerekmez.
  proof_title: Vaat değil, ölçüm
  proof_lead: >-
    Bu sitedeki her iddia tek bir başvuru makinesinde çalıştırıldı: Arc iGPU'lu ve Intel AI
    Boost NPU'lu bir Intel Core Ultra 7 268V (Lunar Lake), Ubuntu 26.04.
  stats:
    - value: "12"
      label: başvuru makinesinde doğrulanan mimari ailesi; Qwen ve Llama'dan Gemma, Granite ve Phi'ye
    - value: "~68 token/s"
      label: NPU'da Qwen3-0.6B, simetrik INT4, <code>yonga bench</code> ile ölçüldü
    - value: "1,4 GiB"
      label: Qwen3-0.6B için indirilen; yalnızca dönüştürmenin okuduğu dosyalar
  proof_link: Test edilen donanım ve modelleri görün
  proof_href: hardware/
  pipeline_title: prepare adım adım ne yapar
  pipeline:
    - Kaynağı inceleme
    - Uyumluluğu çözümleme
    - Ön denetim
    - İndirme
    - Dönüştürme
    - Doğrulama / onarım
    - Çalışma zamanını yoklama
  pipeline_note: Her adım, <code>yonga logs</code> ile geri okuyabileceğiniz bir işe kaydedilir.
  cta_title: Denemeye hazır mısınız?
  cta_text: yonga'yı kurun, <code>yonga doctor</code> çalıştırın ve ilk modelinizi birkaç dakikada hazırlayın.
  cta_primary: Hızlı başlangıcı okuyun
  cta_secondary: Belgelere göz atın
  cta_secondary_link: concepts/
  disclaimer: yonga bağımsız bir açık kaynak projesidir; Intel ile bağlantılı değildir ve Intel tarafından onaylanmamıştır. GNU AGPL v3.0 veya sonrası ile lisanslanmıştır.
---
