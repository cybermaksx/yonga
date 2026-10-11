# Gizlilik ve güvenlik

yonga modellerinizi kendi makinenizde çalıştırır ve onu nasıl kullandığınıza dair hiçbir şey
göndermez. Bu sayfa ağı kullandığı her durumu, Hugging Face erişim belirtecinizi nasıl ele aldığını,
bir model deposundaki kodu ne zaman çalıştıracağını, indirdiklerinizi nasıl denetleyebileceğinizi,
yerel sunucuyu nasıl gizli tutacağınızı ve bir güvenlik açığını nasıl bildireceğinizi listeler.
Güvenlik politikasının kendisi GitHub'daki
[SECURITY.md](https://github.com/ooguz/yonga/blob/main/SECURITY.md) belgesidir.

## Telemetri yok { #telemetry }

yonga telemetri içermez: kullanım istatistiği, çökme raporu ya da güncelleme denetimi yoktur.
Bir `telemetry` ayarı yalnızca açılmaya çalışıldığında reddedilebilsin diye vardır. Yapılandırma
dosyasındaki `telemetry = true` ya da true olarak tanımlanmış `YONGA_TELEMETRY`, yonga'yı
`config.telemetry.rejected` ile durdurur.

yonga, OpenVINO'nun da onun adına rapor göndermesini engeller. OpenVINO'nun dönüştürme API'si her
içe aktarılışını `~/intel/stats` içinde sayar ve `~/intel` içinde bir onay dosyası yoksa bunu
`openvino_telemetry` paketi aracılığıyla Google Analytics'e bildirir. yonga, başlattığı her Python
sürecinde bu paketin içe aktarılmasını olanaksız kılar; yönetilen dışa aktarma ortamında çalışan
dışa aktarıcı da buna dahildir. Hiçbir şey gönderilmez, sayaca dokunulmaz ve yonga bunu ev dizininize
bir vazgeçme (opt-out) dosyası yazmadan yapar.

!!! note "Not"
    `~/intel/stats` yine de değişiyorsa, OpenVINO'yu başka bir program içe aktarmıştır; örneğin düz
    bir `python -c "import openvino"`. Bu yüzden yonga, bir içe aktarmanın denetlenmesi gerektiğinde
    raporlamayı kapalı tutan `yonga doctor --full` komutunu önerir.

## yonga ağı ne zaman kullanır { #network }

yonga yalnızca bir komut ihtiyaç duyduğunda ağ isteği yapar:

| Çalıştırdığınız | Bağlandığı yer | Neden |
|---|---|---|
| `search`, `inspect`, `prepare` | Hugging Face Hub | Model meta verisi; `prepare` için ayrıca model dosyaları. `prepare` yalnızca dönüştürmenin okuduğu dosyaları indirir. |
| `doctor`, bir token tanımlıyken | Hugging Face Hub | Token'ın geçerli olduğunu denetlemek ve ait olduğu hesabı göstermek için. |
| `prepare`, bir yönetilen dışa aktarma ortamının kurulması gerektiğinde | PyPI ve PyTorch'un CPU dizini (`download.pytorch.org`) | Dönüştürme araç zincirini `~/.cache/yonga/envs` altına kurmak için. Plan önce onayınızı ister. |
| `registry update` | Verdiğiniz `https://` adresi | İmzalı bir kayıt defteri paketini ve imzasını almak için. yonga'nın varsayılan bir adresi yoktur. |
| `publish` | Hugging Face Hub | Yayımlamayı seçtiğiniz bir hazırlanmış modeli (artifact) yüklemek için. |

`chat`, `probe`, `bench`, `eval` ve `serve`, diskinizdeki hazırlanmış modelden çalışır ve hiçbir
yere bağlanmaz. `serve` istekleri dinler ama bir isteğin adlandırdığı hiçbir şeyi indirmez
([Yerel sunucu](#serve) bölümüne bakın).

### Çevrimdışı kip { #offline-mode }

yonga'nın Hub'a hiç bağlanmadığından emin olmak için genel `--offline` seçeneğini kullanın ya da
`YONGA_OFFLINE` veya `HF_HUB_OFFLINE` tanımlayın:

```bash
yonga --offline prepare Qwen/Qwen3-0.6B --device cpu
```

`inspect` ve `prepare` bu durumda yalnızca yerel Hugging Face önbelleğinden çalışır. `search`
reddeder, çünkü arama bir ağ isteğidir; önbellekte olmayan bir model ise `hub.offline.missing` ile
başarısız olur. Var olan bir yönetilen dışa aktarma ortamı çevrimdışıyken de yeniden kullanılır,
ama yenisi kurulamaz.

## Hugging Face erişim belirteciniz { #tokens }

Erişim belirtecine (token) yalnızca erişimi kısıtlı (gated) ya da özel modeller için ihtiyacınız
olur. yonga hiçbir zaman token istemez ve saklamaz: `huggingface_hub` kütüphanesinin bulduğu
token'ı, yani `HF_TOKEN` ortam değişkenindekini ya da `hf auth login` ile açılmış oturumunkini
kullanır.

Token yonga'nın çıktısında hiçbir zaman görünmez:

- İş günlükleri, tanılar ve `--debug` çıktısı, token'a benzeyen metni ve `HF_TOKEN` gibi
  değişkenlerin değerini `***REDACTED***` ile değiştiren bir maskeleme süzgecinden geçer.
- Önerilen bir komut sizin komut satırınızı tekrarladığında, kimlik bilgisi taşıyan seçenekler
  çıkarılır ve gizli görünen değerler maskelenir.
- `doctor` yalnızca oturum açıp açmadığınızı, hesap adını ve kimlik bilgisinin nerede bulunduğunu
  (örneğin `credential source: environment:HF_TOKEN`) gösterir; token'ı asla göstermez.
- Hata raporu paketi (`yonga logs JOB_ID --bundle`) aynı şekilde maskelenir; ayrıca ev dizininizi,
  ana makine adını, oturum adınızı ve Hugging Face hesap adınızı anonimleştirir. Bir yere eklemeden
  önce okuyun.

## Model depolarındaki kod { #remote-code }

Bazı depolar kendi Python modelleme kodunu içerir; Transformers bu kodu yalnızca
`trust_remote_code` ile çalıştırır. yonga bunu hiçbir zaman kendiliğinden etkinleştirmez:

- Bir model ancak deposundaki kod çalıştırılarak dönüştürülebiliyorsa, `prepare` hiçbir şey
  yapmadan `source.remote_code_required` ile (çıkış kodu 22) durur.
- Uzak koda `--trust-remote-code` ile izin verirsiniz. O zaman bile yonga terminalde sorar ve
  **`--yes` bu soruyu asla yanıtlamaz**. Terminalsiz bir oturumda yanıt hayırdır.
- Kayıt defteri, mimariyi Transformers'ın kendisinin uyguladığını biliyorsa (Phi-4-mini bir
  örnektir), deponun kodu yok sayılır ve hiç çalıştırılmaz; plan da bunu söyler.

Bu seçeneği vermeden önce deponun kodunu inceleyin. O kod, kullanıcınızın izinleriyle çalışır.

## Uyumluluk kayıt defteri { #registry }

Yerleşik uyumluluk kayıt defteri (registry) yonga'nın içinde gelir. Sürümler arasında daha yeni bir
**uzak paket** ekleyebilirsiniz; yonga onu doğrulanana kadar güvenilmez sayar:

- Yalnızca `yonga registry update` çalıştırdığınızda, bir `https://` adresinden ya da diskteki bir
  dosyadan alınır.
- **Sizin** `~/.config/yonga/registry-keys/` içine koyduğunuz bir açık anahtarla doğrulanan bir
  Ed25519 imzası taşımalıdır. Varsayılan olarak hiçbir anahtara güvenilmez.
- Her yüklemede yeniden denetlenir: süresi dolmamış, kurulu olandan eski değil, daha yeni bir yonga
  için yapılmamış ve kayıt defterinin katı şemasına uyan düz YAML verisi olmalıdır.
- Yalnızca yonga'nın içinde zaten bulunan onarımları ve araç çağırma ayrıştırıcılarını seçebilir. Kayıt
  defteri verisi asla kod getiremez.

`yonga registry status` kurulu paketi ve hâlâ doğrulanıp doğrulanmadığını gösterir,
`yonga registry reset` ise onu kaldırır. Ayrıntılar
[kayıt defteri belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/REGISTRY_SPEC.md#121-remote-bundles-implemented)
yer alır.

## İndirdiklerinizi denetlemek { #downloads }

Her [GitHub sürümü](https://github.com/ooguz/yonga/releases) AppImage'ı bir `.sha256` dosyasıyla
birlikte sunar. AppImage'ı çalıştırmadan önce denetleyin:

```bash
sha256sum --check yonga-0.5.1-x86_64.AppImage.sha256
```

Sürüm ayrıca `yonga-0.5.1-x86_64.AppImage.packages.txt` dosyasını da içerir. Bu dosya paketlenmiş
Python yorumlayıcısını, SHA-256 değeriyle birlikte yonga wheel'ini ve imajdaki her Python paketinin
tam sürümünü listeler. Aynı liste imajın içinde `usr/share/yonga/packages.txt` olarak da bulunur.
İmajı oluşturan betik yorumlayıcıyı ve AppImage araçlarını SHA-256 ile sabitler; Python paketleri
ise imaj oluşturulurken PyPI'dan çözümlenir, her sürümün onları kaydetmesinin nedeni de budur.

## Yerel sunucu { #serve }

`yonga serve`, diğer programlara modelinize OpenAI uyumlu bir API sunar. Varsayılan olarak yalnızca kendi makinenizden erişilebilir:

- `127.0.0.1` üzerinde (8000 numaralı bağlantı noktası) dinler, yani ona yalnızca makinenizdeki
  programlar ulaşabilir. `--host` adresi değiştirir. Adres bir geri döngü (loopback) adresi değilse
  yonga, modele başka makinelerden ulaşılabileceği uyarısını yazdırır; anahtar tanımlı değilse,
  ona ulaşabilen herkesin onu kullanabileceğini de ekler.
- Anahtar zorunlu kılmak için sunucuyu başlatmadan önce `YONGA_API_KEY` tanımlayın. Bu durumda
  `/health` dışındaki her uç nokta `Authorization: Bearer <key>` ister, yoksa 401 ile yanıt verir.
  Anahtar yalnızca ortamdan okunur, hiçbir zaman bir komut satırı seçeneğinden okunmaz; asla
  günlüğe yazılmaz ya da yazdırılmaz.
- Bir isteğin içerdiği URL'yi asla açmaz. Görüntüler yalnızca isteğe gömülü olarak, `data:image/png`,
  `jpeg`, `webp` ya da `gif` base64 URL'leri biçiminde kabul edilir; diğer her URL reddedilir.
- Modelin yazdığı araç çağrıları istemcinize döndürülür ve onları çalıştırıp çalıştırmayacağına
  istemci karar verir. Sunucu onları asla çalıştırmaz.

```bash
export YONGA_API_KEY="$(openssl rand -hex 32)"
yonga serve ARTIFACT_ID --host 0.0.0.0
```

`--host` bir geri döngü adresi olmadığında her zaman bir anahtar tanımlayın. API'nin kullanımı için
[OpenAI uyumlu sunucu](guide/serve.md) sayfasına bakın.

## yonga makinenizde neleri tutar { #local-data }

yonga'nın yazdığı her şey kendi dizinlerinde kalır (`yonga registry paths` ile görebilirsiniz):

- hazırlanmış modeller, iş günlükleri ve performans ölçümü sonuçları `~/.local/share/yonga` altında;
- `chat` içinde yazdığınız istemler (prompt), artifact başına bir dosya olan bir satır düzenleme
  geçmişi olarak `~/.local/share/yonga/chat-history` altında;
- geçici dosyalar, derleme önbelleği ve yönetilen dışa aktarma ortamları `~/.cache/yonga` altında;
- ayarlarınız ve kayıt defteri geçersiz kılma (override) dosyalarınız `~/.config/yonga` altında.

Kaynak modeller olağan Hugging Face önbelleğinde kalır ve asla değiştirilmez. yonga hiçbir zaman
`sudo` çalıştırmaz, sistem ayarlarını değiştirmez ve kendisini kurduğunuz Python ortamına paket
kurmaz. Her şeyin nasıl kaldırılacağını [SSS](faq.md#uninstall) anlatır.

## Güvenlik açığı bildirmek { #vulnerabilities }

Bir güvenlik açığını GitHub güvenlik duyurusu (security advisory) aracılığıyla gizli olarak
bildirin: <https://github.com/ooguz/yonga/security/advisories/new>. Lütfen bunun için herkese açık
bir konu (issue), çekme isteği (pull request) ya da tartışma açmayın.

Hangi sürümün ve kurulum yönteminin (pip ya da AppImage) etkilendiğini, sorunun nasıl yeniden
üretileceğini ve bir saldırganın ne kazandığını belirtin. Yanıt duyurunun içinde gelir ve
doğrulanan bir sorun yeni bir yama sürümünde düzeltilir. Yalnızca en yeni sürüm düzeltme alır.

Kapsamda yonga'nın kendi kodu ve birlikte gelenler vardır: `serve` API'si, token'ların ele alınışı, uzak
kod onayı, imzalı kayıt defteri paketleri, hata raporu paketindeki maskeleme ve sürüm dosyaları. OpenVINO,
Transformers, PyTorch ve yonga'nın kurduğu diğer paketlerdeki ya da modellerin kendisindeki
sorunları en iyisi kendi geliştiricilerine bildirmektir.
