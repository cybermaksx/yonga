# SSS

yonga hakkında en sık sorulan sorulara kısa yanıtlar ve ayrıntıya giren sayfaların bağlantıları.
Sorunuz bir hata iletisiyle ilgiliyse, başlamak için [Sorun giderme](troubleshooting.md) sayfası
daha uygundur.

## yonga'nın Intel ile bir bağlantısı var mı? { #intel }

Hayır. yonga, GNU Affero Genel Kamu Lisansı v3.0 veya sonrası (AGPL-3.0-or-later) ile lisanslanmış bağımsız bir açık kaynak projesidir. Intel'in açık kaynaklı
OpenVINO araç takımı (OpenVINO, OpenVINO GenAI, Optimum Intel ve NNCF) ile Hugging Face Hub
üzerine kuruludur, ama Intel tarafından geliştirilmez, onaylanmaz ya da desteklenmez.

## Hangi modeller çalışır? { #models }

yonga, OpenVINO GenAI'nin çalıştırabildiği metin üretme modelleri (LLM) ve görsel-dil modelleri
(VLM) içindir. Şunları hazırlayabilir:

- Transformers ağırlıkları (safetensors) içeren Hugging Face depoları; bunları dönüştürür ve
  nicemleme (quantization) uygular;
- zaten bir OpenVINO modeli içeren depolar; bunları olduğu gibi kullanır;
- llama, qwen2 ve qwen3 mimarilerinin GGUF dosyaları; bunları içe aktarır.

Gerçek donanımda test edilen aileler ve aygıt başına sonuçları
[Test edilen donanım ve modeller](hardware.md) sayfasında listelenir. Yaygın ailelerin diğer
modelleri genel kurallarla ele alınır ve `tested` yerine `inferred` olarak işaretlenir. Hiçbir şey
indirmeden önce öğrenmek için yonga'ya sorun:

```bash
yonga search qwen3
yonga inspect Qwen/Qwen3-0.6B
```

`inspect` yalnızca modelin meta verisini okur ve makinenizdeki her aygıt için modelin çalışmasının
beklenip beklenmediğini ve uyumluluk kayıt defterinin (registry) bundan ne kadar emin olduğunu
gösterir.

## yonga neden varsayılan olarak INT4 kullanıyor? { #int4 }

Daha küçük ağırlıklar daha az disk ve daha az bellek kaplar; bu bir dizüstü bilgisayarda önemlidir.
Örneğin `Qwen/Qwen3-0.6B` için 1,4 GiB indirilir; NPU için hazırlandıktan sonra 387 MiB yer kaplar.
Varsayılan profillerin hepsi ağırlıkların çoğunu 4 bitlik tamsayılar olarak saklar.

INT4 nicemleme çıktı kalitesinden bir miktar götürür ve yonga'nın varsayılanları bu
bedelin ölçümlerine göre seçilir. 5B parametrenin altında CPU ve GPU varsayılanları katmanların
beşte birini INT8'de tutar, NPU varsayılanı ise grup bazlı INT4 kullanır. On iki model üzerinde
ölçüldüğünde, bu katmanları INT8'de tutmak çıktıyı INT8 referansına (KL ıraksaması ile) %16-42 daha
yakınlaştırdı; bedeli %9-18 daha fazla disk oldu. Sayılar
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#10-small-model-defaults-twelve-models-three-devices)
belgesindedir.

`--profile` ile başka bir profil seçebilirsiniz. `cpu-int8-asym-ref` ve `cpu-fp16-ref` CPU'da daha
fazla hassasiyet korur; `yonga eval` ise farkı kendi modelinizde ölçer:

```bash
yonga registry profiles
yonga prepare Qwen/Qwen3-0.6B --device cpu --profile cpu-int8-asym-ref
```

## yonga neden NPU'yu değil de GPU'yu seçti? { #why-gpu }

`--device auto` ile (varsayılan) yonga aygıtları önce kayıt defterinin her aygıtın bu modelle
uyumu hakkında bildiklerine göre sıralar; bunlar eşitse önce GPU'yu, sonra NPU'yu, sonra CPU'yu
tercih eder. Bir Intel AI PC'de tümleşik GPU en geniş model yelpazesini kabul eder, NPU ise
en seçici olanıdır. Yerleşik kayıt defterindeki her aile de GPU'yu önerir. Planın "Device decision"
satırı hangi kuralın karar verdiğini söyler.

NPU'yu kullanmak için onu isteyin:

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu
```

Açıkça istenen bir aygıt hiçbir zaman sessizce değiştirilmez: NPU modeli çalıştıramazsa `prepare`
durur ve bunun yerine ne çalıştırmanız gerektiğini söyler. Kuralları
[yonga nasıl çalışır](concepts.md#devices) sayfası açıklar.

## yonga NPU olmadan çalışır mı? { #no-npu }

Evet. yonga ilk olarak NPU için yapıldı, ama yalnızca onun için değil. CPU her x86_64 makinede
çalışır; tümleşik Intel GPU ise Intel'in hesaplama çalışma ortamı (compute runtime) kurulduktan
sonra çalışır (Ubuntu'da `intel-opencl-icd`). `yonga doctor`, OpenVINO'nun makinenizde hangi
aygıtları görebildiğini gösterir.

## Windows ya da macOS'ta çalışır mı? { #windows }

Henüz değil. yonga yalnızca x86_64 üzerinde Linux'u destekler. Aygıt, sürücü ve yol denetimleri
Linux için yazılmış ve test edilmiştir; tasarım ileride Windows'a kapıyı açık tutar. AppImage
glibc 2.28 ya da daha yenisini, pip kurulumu ise Python 3.11 ile 3.14 arasını gerektirir.

## Modeller nerede saklanır ve ne kadar disk gerekir? { #storage }

| Ne | Nerede |
|---|---|
| İndirilen kaynak modeller | Olağan Hugging Face önbelleği, varsayılan olarak `~/.cache/huggingface/hub` |
| Hazırlanmış modeller (artifact'ler), günlükler, performans ölçümleri | `~/.local/share/yonga` |
| Geçici dönüştürme dosyaları, derleme önbelleği, yönetilen dışa aktarma ortamları | `~/.cache/yonga` |
| Ayarlar ve kayıt defteri geçersiz kılma (override) dosyaları | `~/.config/yonga` |

`yonga registry paths` gerçek konumları yazdırır, `yonga list` ise her artifact'in boyutunu
gösterir. Referans makineden bazı rakamlar:

- `Qwen/Qwen3-0.6B`: 1,4 GiB indirme, 387 MiB'lık bir NPU artifact'i (`npu-int4-sym-g128`) ve
  172 MiB'lık bir NPU derleme önbelleği girdisi.
- 9B'lik bir model: yaklaşık 18 GiB kaynak ağırlık, artı dönüştürülürken yaklaşık 44 GiB geçici
  alan.
- Bir yönetilen dışa aktarma ortamı: yaklaşık 3 GiB'a kadar; bir kez kurulur ve yeniden kullanılır.

`prepare` ihtiyaç duyduğu alanı hesaplar ve sığmıyorsa indirmeden önce reddeder. Başka bir disk
kullanmak için `YONGA_DATA_DIR` ve `YONGA_CACHE_DIR` tanımlayın; temizlik için [Bakım](guide/maintain.md)
sayfasına bakın.

## Zaten OpenVINO biçiminde olan modelleri kullanabilir miyim? { #openvino-models }

Evet, Hugging Face Hub'dan. Bir depo zaten bir OpenVINO modeli içeriyorsa `prepare` onu dönüştürmek
yerine doğrudan kullanır, sonra diğer her artifact gibi doğrular ve yoklar. `inspect` bunu önceden
söyler:

```text
Direct OpenVINO source  yes  OpenVINO/Qwen3-0.6B-int4-ov
```

`prepare` bir Hugging Face depo kimliği alır; bu yüzden yonga'nın hazırlayabilmesi için modelin
Hub'da olması (herkese açık ya da token'ınızla erişilebilen özel bir depoda) gerekir.

## yonga, Ollama ya da llama.cpp ile nasıl karşılaştırılır? { #ollama }

Benzer sorunları farklı yollarla çözerler. Ollama ve llama.cpp, GGUF biçimi etrafında kurulmuş genel
amaçlı yerel model çalıştırıcılarıdır. yonga'nın kendi çıkarım motoru yoktur: modelleri Intel'in
OpenVINO'su için hazırlar ve Intel NPU, GPU ya da CPU üzerinde OpenVINO GenAI ile çalıştırır; bu
sırada verdiği aygıt ve nicemleme kararlarını da açıklar.

`serve` komutu OpenAI uyumlu bir API sunar, böylece bu API için yazılmış istemciler ikisini de
kullanabilir. Bilerek küçük tutulmuştur: tek model, aynı anda tek istek. Başka bir araç sizin için
zaten iyi çalışıyorsa geçiş yapmanıza gerek yok. yonga, bir Intel AI PC'nin NPU'sunda ve GPU'sunda
OpenVINO aracılığıyla model çalıştırmak için tasarlanmıştır.

## yonga herhangi bir yere veri gönderiyor mu? { #data }

Hayır. Telemetri yoktur ve yonga OpenVINO'nun kendi kullanım raporlamasını da engeller. Hugging Face
Hub'a yalnızca bir komut ihtiyaç duyduğunda (arama, inceleme, indirme, yayımlama ya da
`doctor`'ın tanımladığınız token'ı denetlemesi), Python paket dizinlerine yalnızca onayladığınız
bir yönetilen dışa aktarma ortamını kurmak için, bir kayıt defteri adresine ise yalnızca o adresle
`registry update` çalıştırdığınızda bağlanır.
`--offline` tüm Hub erişimini durdurur. Her ağ bağlantısını [Gizlilik ve güvenlik](privacy.md)
sayfası listeler.

## Bir Hugging Face hesabına ihtiyacım var mı? { #account }

Herkese açık modeller için hayır. Hesaba ve bir erişim belirtecine (token) yalnızca erişimi kısıtlı
(gated) ya da özel depolar için ihtiyacınız olur: modelin Hub sayfasında koşullarını kabul edin,
sonra `hf auth login` ile oturum açın ya da token'ı `HF_TOKEN` olarak dışa aktarın.

## yonga'yı nasıl güncellerim? { #update }

=== "AppImage"

    Yeni AppImage'ı ve `.sha256` dosyasını [sürümler sayfasından](https://github.com/ooguz/yonga/releases)
    indirin, denetleyin ve eskisinin üzerine kurun. `X.Y.Z` yerine yeni sürümü yazarak:

    ```bash
    sha256sum --check yonga-X.Y.Z-x86_64.AppImage.sha256
    chmod +x yonga-X.Y.Z-x86_64.AppImage
    ./yonga-X.Y.Z-x86_64.AppImage self install
    ```

    `self install` iki sürümü de gösterir ve eski kopyayı değiştirmeden önce sorar.

=== "pip"

    yonga'yı kurduğunuz sanal ortamda:

    ```bash
    pip install --upgrade "yonga[openvino,export]"
    ```

Hazırlanmış modelleriniz korunur. Yeni sürüm daha yeni bir OpenVINO getiriyorsa eski derlenmiş
blob'lar `stale-runtime` olur ve `yonga cache prune` onları kaldırır.

## Her şeyi nasıl kaldırırım? { #uninstall }

1. Taşımış olma ihtimalinize karşı yonga'nın dosyalarını nerede tuttuğuna bakın:

    ```bash
    yonga registry paths
    ```

2. Programı kaldırın.

    === "AppImage"

        ```bash
        yonga self uninstall
        ```

        Bu, `~/.local/bin` içindeki kopyayı kaldırır. İndirdiğiniz AppImage dosyasını da silin.

    === "pip"

        ```bash
        pip uninstall yonga
        ```

        Ya da sanal ortamın tamamını silin.

3. yonga'nın verilerini, önbelleğini ve ayarlarını kaldırın (bunlar varsayılan konumlardır):

    ```bash
    rm -rf ~/.local/share/yonga ~/.cache/yonga ~/.config/yonga
    ```

4. Kaynak modeller, başka araçların da kullandığı ortak Hugging Face önbelleğindedir. Artık
   ihtiyacınız olmayanları Hugging Face'in `hf` komutuyla kaldırın:

    ```bash
    hf cache ls
    hf cache rm model/Qwen/Qwen3-0.6B
    ```

yonga bu konumların dışına hiçbir şey kurmaz ve sistem ayarlarını değiştirmez; bu yüzden geride
başka bir şey kalmaz.
