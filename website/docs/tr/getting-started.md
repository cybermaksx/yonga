# Hızlı başlangıç

Bu sayfa sizi dört adımda sıfırdan yerel bir sohbete götürür: yonga'yı kurun, makinenizi denetleyin,
küçük bir model hazırlayın ve onunla konuşun. Örnek olarak tüm yolu birkaç dakikada tamamlayacak
kadar küçük bir model olan `Qwen/Qwen3-0.6B` kullanılır. Tüm seçenekler ve her adımın ayrıntıları
için [Kurulum](installation.md) sayfasına ve [kılavuzlara](guide/prepare.md) bakın.

## Başlamadan önce { #before-you-start }

Gerekenler:

- x86_64 üzerinde Linux. Windows henüz desteklenmiyor.
- İlk hazırlama için internet bağlantısı: yonga modeli Hugging Face Hub'dan indirir (bu model için
  yaklaşık 1,4 GiB).
- Ev dizininizde birkaç GiB boş disk alanı. Hazırlanmış model (artifact) yaklaşık 430 MiB tutar; ilk dönüştürme
  ayrıca en fazla yaklaşık 3 GiB'lık bir yönetilen dışa aktarma ortamı kurabilir (bkz.
  [3. adım](#prepare)).

yonga'nın kendisi root erişimi istemez: hiçbir zaman `sudo` çalıştırmaz ve her şeyi ev dizininizin
altında tutar. Makinenizde henüz yapılmadıysa GPU ya da NPU için bir kerelik sürücü ve `render`
grubu ayarı `sudo` gerektirir; bu komutları kendiniz çalıştırırsınız (bkz.
[Kurulum](installation.md#the-intel-npu)).

## 1. yonga'yı kurun { #install }

=== "AppImage"

    AppImage, içinde yonga, kendi Python'u ve OpenVINO çalışma zamanı bulunan tek bir dosyadır.
    Dosyayı sağlama toplamıyla birlikte indirin, doğrulayın ve `yonga` komutu olarak kurun:

    ```bash
    curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage
    curl -LO https://github.com/ooguz/yonga/releases/download/v0.5.1/yonga-0.5.1-x86_64.AppImage.sha256
    sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
    chmod +x yonga-0.5.1-x86_64.AppImage
    ./yonga-0.5.1-x86_64.AppImage self install
    ```

    `self install` dosyayı `~/.local/bin/yonga` konumuna kopyalar. Bu dizin henüz `PATH` içinde
    değilse bunu söyler ve çalıştırmanız gereken satırı yazdırır. Ubuntu veya Debian'da ilk
    kurulumdan sonra bir kez oturumu kapatıp yeniden açmak genellikle yeterlidir.

=== "pip"

    PyPI'dan bir sanal ortama kurun (Python 3.11 ile 3.14 arası). NVIDIA GPU'su olmayan bir Intel
    makinede önce `torch` ve `torchvision` paketlerinin CPU derlemelerini birlikte kurun: böylece
    hiç kullanmayacağınız yaklaşık 4 GB'lık CUDA kitaplıklarını indirmekten kurtulursunuz.

    ```bash
    python3 -m venv ~/.venvs/yonga
    source ~/.venvs/yonga/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install "yonga[openvino,export,serve]"
    ```

    `yonga` komutu, ortam etkin olduğu sürece kullanılabilir.

Her ek paketin (extra) ne işe yaradığı, AppImage için FUSE notu, yükseltme ve kaldırma için
[Kurulum](installation.md) sayfasına bakın.

## 2. Makinenizi denetleyin { #check }

```bash
yonga doctor
```

`doctor`; işletim sisteminizi, Python'u, OpenVINO paketlerini, OpenVINO'nun gördüğü her aygıtı (CPU,
GPU, NPU), NPU'yu kullanma izniniz olup olmadığını, boş disk alanını ve Hugging Face kimlik
doğrulamasını raporlar. Hiçbir şeyi değiştirmez; bir şey eksikse kendiniz çalıştırmanız gereken
komutu söyler.

Sonucu çıkış kodu özetler:

| Çıkış kodu | Anlamı |
|---|---|
| `0` | Her şey kullanılabilir. |
| `2` | Kullanılabilir, ama uyarılar var: rapor `Usable, with warnings above.` satırıyla biter. `!` ile başlayan satırları okuyun. |
| `10` | yonga'nın aşamayacağı bir eksik var; rapor neyin eksik olduğunu söyler. |

!!! note "Not"
    Ubuntu'da `/tmp` RAM üzerinde duran küçük bir dosya sistemidir, bu yüzden `doctor` genellikle
    **Storage → system temp** altında bir uyarıyla `2` koduyla çıkar. Bu uyarıyı göz ardı
    edebilirsiniz: yonga modelleri kendi disk tabanlı dizininde dönüştürür ve `/tmp`'nin
    büyütülmesine hiç ihtiyaç duymaz.

**NPU** bölümünde bir uyarı görürseniz [Kurulum](installation.md#the-intel-npu) sayfasına bakın.

## 3. Küçük bir model hazırlayın { #prepare }

```bash
yonga prepare Qwen/Qwen3-0.6B
```

yonga modelin üst verisini (metadata) okur, bir aygıt ve bir nicemleme (quantization) profili seçer ve bir
**hazırlama planı** yazdırır: tam revizyon, aygıt ve neden seçildiği, profil, beklenen indirme
boyutu, gereken disk alanı ve kurması gereken her şey. Uzun sürecek ya da çok yer kaplayacak bir
işe girişmeden önce onayınızı ister. Onay verdiğinizde modeli indirir, dönüştürür, bilinen
sorunları onarır ve modele hazır demeden önce aygıt üzerinde kısa bir metin üretme testi çalıştırır.

Aygıtı kendiniz seçmek için `--device npu`, `--device gpu` veya `--device cpu` ekleyin. Aygıtı
açıkça belirttiğinizde yonga kendiliğinden başka bir aygıta geçmez. Aşağıda Intel Core Ultra 7 268V
üzerinde `--device npu` ile yapılmış bir çalıştırmanın kısaltılmış çıktısı var:

```text
╭───────────────────────────── Preparation plan ─────────────────────────────╮
│                                                                            │
│  Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814  │
│                              85cdf75e47ca                                  │
…
│  Device decision             NPU  supported (tested)  NPU was requested    │
│                              explicitly; yonga never switches device on    │
│                              its own.                                      │
│  Profile                     npu-int4-sym-g128                             │
│                              NPU group-wise INT4; the NPU default below    │
│                              5B parameters                                 │
│  Quantization                int4, group size 128, ratio 1, symmetric      │
…
│  Expected download           1.4 GiB  only the files the conversion reads  │
…
│  Estimated artifact          about 430 MiB                                 │
…
│  Export environment          managed export-fbfac3ac91ba                   │
│                              transformers==5.5.4  will be built            │
…
╰───────────────── job de862650-85f5-44f4-9921-1a415bc541d3 ─────────────────╯
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
Building managed export environment export-fbfac3ac91ba (transformers==5.5.4)
4/7 Downloading
Fetching Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca (1.4 GiB)
5/7 Converting
Converting with profile npu-int4-sym-g128
6/7 Validating / repairing
Inventorying OpenVINO IR
Applying registry repairs: ov-tokenizer-precision-string-casing
7/7 Probing runtime
· NPU  Compiling openvino_model.xml on NPU
· NPU  Generating 24 tokens on NPU
…
Ready
  Artifact  cc24636a2f49356a
…
  Device    NPU
  Profile   npu-int4-sym-g128
```

Sonuç bir **artifact**'tir: dönüştürülmüş model ve onun nereden geldiğini, nasıl üretildiğini
kaydeden bir manifest. `Artifact` satırındaki değer onun kimliğidir; `yonga list` bu
makinedeki tüm artifact'leri gösterir.

!!! tip "Yönetilen dışa aktarma ortamı"
    AppImage ve `export` ek paketi olmadan yapılmış bir pip kurulumu dönüştürme araç zincirini
    içermez. Bu yüzden ilk dönüştürme, yukarıdaki planda yazdığı gibi, `~/.cache/yonga/envs` altında
    bir *yönetilen dışa aktarma ortamı* kurar; sonraki hazırlamalar onu yeniden kullanır. Kendi
    Python ortamınız hiçbir zaman değiştirilmez. Nedenini
    [Kurulum](installation.md#managed-export-environments) sayfası açıklıyor.

## 4. Modelle sohbet edin { #chat }

```bash
yonga chat Qwen/Qwen3-0.6B
```

Model kimliğini ya da `prepare` komutunun yazdırdığı artifact kimliğini verebilirsiniz
(`yonga chat cc24636a2f49356a`). `--device` verilmezse `chat`, artifact'in hazırlandığı aygıtı
kullanır. Oturum bir özetle açılır ve her yanıt bir ölçüm satırıyla biter:

```text
Loading the model on NPU...
╭──────────────────────────────── Local Chat ────────────────────────────────╮
│                                                                            │
│  Model              cc24636a2f49356a                                       │
│  Source             Qwen/Qwen3-0.6B@c1899de289a0                           │
│  Profile            npu-int4-sym-g128                                      │
│  Device             NPU                                                    │
│  Max output         4096 tokens                                            │
│  Reasoning display  hidden                                                 │
…
│  Thinking mode      off (enable_thinking=false)                            │
│  Sampling           temp 0.7 • top-p 0.8 • top-k 20                        │
│  Metrics footer     on                                                     │
│                                                                            │
╰──────────────────────────── /help for commands ────────────────────────────╯
…
TTFT 391 ms • 44.3 tok/s • TPOT 22.6 ms/token • 22 → 27 tokens
```

Alt satır; ilk token'a kadar geçen süreyi, üretim hızını, çıktı token'ı başına süreyi ve istem ile
yanıtın token sayılarını gösterir. Sohbet komutları (`/reset`, `/tokens N`, `/temp X`,
`/reasoning on|off` ve diğerleri) için `/help` yazın; çıkmak için `/exit` yazın ya da ++ctrl+d++
tuşlarına basın.

## Sonraki adımlar { #next-steps }

- **Modeli başka araçlara sunun.** `yonga serve Qwen/Qwen3-0.6B`, `http://127.0.0.1:8000/v1`
  adresinde OpenAI uyumlu bir API başlatır. Bkz. [OpenAI uyumlu sunucu](guide/serve.md).
- **Ölçün.** `yonga bench Qwen/Qwen3-0.6B` bir istem takımı çalıştırır ve performans ölçümünün
  sonucunu JSON olarak kaydeder; `yonga probe` bir artifact'in bir aygıtta derlenip derlenmediğini
  yoklar; `yonga eval` nicemlemenin modelden neler götürdüğünü ölçer. Bkz.
  [Sohbet, yoklama, ölçüm ve değerlendirme](guide/run.md).
- **Başka bir model hazırlayın.** `yonga inspect MODEL`, hiçbir şey indirilmeden önce bir modeli
  hazırlamanın neler gerektireceğini söyler. Bkz. [Model hazırlama](guide/prepare.md) ve
  [Test edilen donanım ve modeller](hardware.md).
- **Bir şey başarısız olursa** mesajda bir tanı kodu ve sonraki adım yer alır. Bkz.
  [Sorun giderme](troubleshooting.md).
