# Model hazırlama

`yonga prepare`, bir Hugging Face deposunu hazırlanmış bir modele (artifact) dönüştürür: indirilmiş,
dönüştürülmüş, doğrulanmış, gerekiyorsa onarılmış ve aygıtınızda derlenip token üretebildiği
gösterilmiş bir OpenVINO modeli. Bu sayfa yolun tamamını anlatır: model bulmak, planı okumak, aygıt
ve profil seçmek, yonga'nın sorduğu soruları yanıtlamak, erişimi kısıtlı (gated), GGUF ve uzak kod
gerektiren depolar gibi özel durumlar, çevrimdışı çalışma, disk alanı ve komutu yeniden
çalıştırdığınızda ya da iptal ettiğinizde ne olduğu.

## İndirmeden önce bakın { #search-and-inspect }

Aday modelleri bulmak için Hub'da arama yapın:

```bash
yonga search "qwen3 0.6b"
```

```text
Model                                  Arch        CPU   GPU   NPU   Format           DL
────────────────────────────────────────────────────────────────────────────────────────
Qwen/Qwen3-0.6B                        qwen3        ✓     ✓     ✓    safetensors   30.6M
Qwen/Qwen3-Embedding-0.6B              qwen3        ✓     ✓     ✓    safetensors    8.5M
…
✓ supported   ✗ unsupported   ! experimental   ? no registry claim
```

Aygıt sütunları, uyumluluk kayıt defterinin (registry) her *mimari* hakkında bildiklerini gösterir;
deponun dosyalarını denetlemez. `?` kayıt defterinin bir iddiası olmadığı anlamına gelir, modelin
çalıştığı anlamına asla gelmez. Yalnızca bir aygıtı görmek için `--device npu`, daha çok ya da daha
az sonuç için `--limit N` (1 ile 100 arası, varsayılan 20) ekleyin.

Ardından istediğiniz depoyu inceleyin. `inspect` yalnızca Hub üst verilerini ve küçük yapılandırma
dosyalarını okur, ağırlık dosyalarını asla indirmez; *bu* deponun *bu* makinede hazırlanıp
hazırlanamayacağını söyler:

```bash
yonga inspect Qwen/Qwen3-0.6B
```

```text
Model         Qwen/Qwen3-0.6B
Revision      c1899de289a0
License       apache-2.0
…
Size          ~1.4 GiB total weights
Parameters    751,632,384  Hub count; includes embeddings and any vision tower

Compatibility on this host
✓  CPU  supported (tested)
✓  GPU  supported (tested)
✓  NPU  supported (tested)

Preparation
Direct OpenVINO source  no
Safetensors source      yes  Qwen/Qwen3-0.6B
Known derivative        no  No derivative source is declared for this repository.
```

`--files` deponun dosyalarını da listeler; `--revision` ise bir dalı, etiketi ya da commit'i inceler.

## Varsayılanlarla hazırlama { #prepare }

Bir modeli hazırlamak için şunu çalıştırın:

```bash
yonga prepare Qwen/Qwen3-0.6B
```

yonga önce bir **hazırlama planı** yazdırır. Plan **Confirmation required** altında bir şey
listeliyorsa yonga, hiçbir şey indirmeden önce her biri için size sorar; liste boşsa iş doğrudan
devam eder. Ardından yedi aşamayı çalıştırır:

```text
1/7 Inspecting source
2/7 Resolving compatibility
3/7 Preflight
4/7 Downloading
5/7 Converting
6/7 Validating / repairing
7/7 Probing runtime
```

Dönüştürücünün hatasız bitmesi işin sonu değildir: sonuç envantere alınır, denetlenir, kayıt
defterinin bildiği bir kusur varsa onarılır, hedef aygıtta derlenir ve token üretmesi sağlanır.
Ancak bundan sonra hazır bir artifact olur.

Aşağıdaki, referans makinede gerçek bir çalıştırmadır (yonga 0.5.0 ile `Qwen/Qwen3-0.6B` modelinin
NPU için hazırlanması, kısaltılmış; çalıştırmada özel `--data-dir` ve `--cache-dir` konumları
kullanıldı, burada `…` ile kısaltıldı):

```text
╭───────────────────────────── Preparation plan ─────────────────────────────╮
│  Requested model             Qwen/Qwen3-0.6B@c1899de289a04d12100db370d814  │
│                              85cdf75e47ca                                  │
│  Resolved source             Qwen/Qwen3-0.6B                               │
│  Revision                    c1899de289a0                                  │
│  Relationship                original                                      │
│  Source format               safetensors                                   │
│  Architecture                qwen3  rule arch-qwen3                        │
│  Task                        text-generation-with-past                     │
│  Device decision             NPU  supported (tested)  NPU was requested    │
│                              explicitly; yonga never switches device on    │
│                              its own.                                      │
│  Profile                     npu-int4-sym-g128                             │
│                              NPU group-wise INT4; the NPU default below    │
│                              5B parameters                                 │
│  Quantization                int4, group size 128, ratio 1, symmetric      │
│  Chosen because              751.6M parameters (Hugging Face Hub count     │
│                              from safetensors headers) is below 5B:        │
│                              npu-int4-sym-g128 is the NPU profile for      │
│                              that size (evidence: tested).                 │
│  Expected download           1.4 GiB  only the files the conversion reads  │
│  Estimated temp requirement  25 GiB  free 341 GiB in …/cache/tmp/de862650… │
│                              System /tmp is too small; the managed temp    │
│                              directory is used.                            │
│  Estimated artifact          about 430 MiB                                 │
│  Artifact directory          …/data/artifacts/Qwen--Qwen3-0.6B/c1899de289a │
│                              0--npu-int4-sym-g128                          │
│  Remote code                 not required                                  │
│  Export environment          managed export-fbfac3ac91ba                   │
│                              transformers==5.5.4  will be built            │
│                                                                            │
│  Expected repairs                                                          │
│  • ov-tokenizer-precision-string-casing                                    │
│  …                                                                         │
│  Confirmation required                                                     │
│  ·  A managed export environment (transformers==5.5.4) will be built       │
│     under …/cache/envs: up to about 3.0 GiB of packages from PyPI and the  │
│     PyTorch CPU index. It is used only by the export, is reused by later   │
│     jobs, and your own Python environment is not changed.                  │
│  ·  This export is expected to need a known artifact repair:               │
│     ov-tokenizer-precision-string-casing                                   │
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
Hashing 12 files
…

Ready
  Artifact  cc24636a2f49356a
  Path      …/data/artifacts/Qwen--Qwen3-0.6B/c1899de289a0--npu-int4-sym-g128
  Device    NPU
  Profile   npu-int4-sym-g128
  note: The system temporary directory /tmp is too small for this conversion;
the managed directory was used instead.

Chat with it: yonga … chat cc24636a2f49356a
```

Son satır, çalıştıracağınız bir sonraki komuttur. Artifact kimliği (burada `cc24636a2f49356a`);
`chat`, `probe`, `bench`, `eval` ve `serve` komutlarına verdiğiniz değerdir; bkz.
[Sohbet, yoklama, ölçüm ve değerlendirme](run.md) ve [OpenAI uyumlu sunucu](serve.md). `yonga list`
hazır olan tüm artifact'leri gösterir.

## Planı okumak { #the-plan }

En sık bakacağınız satırlar:

| Satır | Size söyledikleri |
|---|---|
| **Requested model**, **Revision** | Depo ve sabitlendiği tam commit. İşteki her indirme bu commit'i kullanır. |
| **Resolved source**, **Relationship** | Gerçekte hangi deponun dönüştürüldüğü. Genellikle istediğiniz depodur (`original`); bildirilmiş bir türev ya da `--source` ile farklı olur. |
| **Device decision** | Aygıt, kayıt defterinin bu aygıt için kararı ve neden seçildiği. |
| **Profile**, **Quantization**, **Chosen because** | Dönüştürme profili ve tek cümleyle neden seçildiği. |
| **Expected download** | Yalnızca dönüştürmenin okuduğu dosyalar; depodaki başka çalışma zamanlarının ağırlıkları ve eğitimden kalan dosyalar atlanır. |
| **Estimated temp requirement** | Dönüştürmenin ihtiyaç duyduğu çalışma alanı ve nerede olacağı. Bkz. [Disk alanı](#disk-space). |
| **Estimated artifact**, **Artifact directory** | Sonucun boyutu ve konumu. |
| **Remote code** | Dönüştürmenin depodaki Python kodunu çalıştırması gerekip gerekmediği. |
| **Export environment** | Dışa aktarmayı hangi Python'un çalıştırdığı: bu ortam ya da yönetilen bir ortam. Bkz. [Yönetilen dışa aktarma ortamları](#managed-environments). |
| **Expected repairs**, **Known limitations** | Kayıt defterinin onarmayı beklediği kusurlar ve bilmeniz gereken diğer her şey. |
| **Confirmation required** | Size sorulacak sorular. |

İş planlandığı gibi tamamlanamayacaksa plan, nedenini söyleyen bir **BLOCKED** bölümü gösterir ve
hiçbir şey indirilmez. Şu durumlar daha hiçbir soru sorulmadan reddedilir: OpenVINO çalışma zamanı
kurulu değil (çıkış kodu 10), kayıt defterinin bu model için dışladığı bir `--device` açıkça
istenmiş (çıkış kodu 50, önerilen aygıtla birlikte) ya da erişimi kısıtlı veya özel bir depo için
Hugging Face erişim belirteci (token) yok (çıkış kodu 21). `--no-managed-env` altında
karşılanamayan dışa aktarıcı gereksinimleri de BLOCKED olarak gösterilir; bu iş sorulardan sonra,
ama yine indirmeden önce, ön denetimde (preflight) durur (çıkış kodu 10). OpenVINO'nun bildirmediği
bir `--device` (çıkış kodu 50) ve `--trust-remote-code` olmadan uzak kod (çıkış kodu 22) da işi
indirmeden önce durdurur, ancak BLOCKED bölümü göstermez. Bkz.
[Çıkış kodları](../reference/exit-codes.md).

## Aygıt seçmek { #device }

`--device` seçeneği `auto` (varsayılan), `npu`, `gpu` ya da `cpu` değerlerinden birini alır.

```bash
yonga prepare Qwen/Qwen3-0.6B --device gpu
```

**Açıkça istenen aygıt asla değiştirilmez.** Aygıt modeli çalıştıramazsa iş durur, nedenini
açıklar ve çalışacak bir aygıt için komutu yazdırır. Çalıştırıp çalıştırmamaya siz karar
verirsiniz. Kayıt defteri ya da OpenVINO aygıtı daha indirmeden önce dışlıyorsa çıkış kodu 50'dir;
model aygıtta derlenirken ortaya çıkan bir hata ise kendi çıkış koduyla biter (örneğin derleme
hatası için 51).

**`--device auto`**, kayıt defterinin bildiklerine ve OpenVINO'nun bu makinede bildirdiği aygıtlara
göre bir aygıt seçer. Kayıt defterinin bu mimari için desteklenmiyor olarak işaretlediği aygıtlar
elenir; kalanlar önce kayıt defterinin kararına, sonra GPU, NPU, CPU sırasına göre dizilir ve bir
mimarinin önerilen aygıtı (resmî ya da test edilmiş kanıta dayanıyorsa) en başa alınır. Seçilen aygıt
yoklamada model ona uymadığı için başarısız olursa auto kipi bir sonraki aygıta geçebilir; geçmeden
önce size sorar. NPU aygıt düğümüne erişim izni olmaması ya da eksik bir sürücü gibi bozuk bir ortam
asla başka bir aygıtın arkasına saklanmaz: düzeltebilmeniz için `prepare` 10 çıkış koduyla durur.
Sorunu görmek için `yonga doctor` çalıştırın.

[yonga nasıl çalışır](../concepts.md) aygıt politikasını daha ayrıntılı anlatır;
[Test edilen donanım ve modeller](../hardware.md) doğrulanmış olanları listeler.

## Profil seçmek { #profile }

Profil, adı olan bir dönüştürme tarifidir: hedef aygıt, ağırlık biçimi ve nicemleme (quantization)
ayarları. yonga varsayılan olarak modelin parametre sayısına göre bir profil seçer:

| Aygıt | 5B parametrenin altında | 5B ve üzeri ya da boyut bilinmiyorsa |
|---|---|---|
| CPU | `cpu-int4-asym-g128-r08` | `cpu-int4-asym-g128` |
| GPU | `gpu-int4-asym-g128-r08` | `gpu-int4-asym-g128` |
| NPU | `npu-int4-sym-g128` | `npu-int4-sym-cw` |

`-r08` profilleri katmanların %20'sini INT8'de tutar. Plandaki **Chosen because** satırı hangi
kuralın karar verdiğini söyler. Modelin `config.json` dosyası, bir profilin 128'lik grup boyutunun
modelin genişliklerini bölemediğini gösteriyorsa o profil indirmeden önce atlanır ve bu satır
nedenini söyler; bkz.
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#54-nncf-refuses-the-profiles-group-size).
Tüm profilleri listelemek için:

```bash
yonga registry profiles
yonga registry profiles --device npu
```

Başka bir profil kullanmak için adını verin:

```bash
yonga prepare Qwen/Qwen3-0.6B --device npu --profile npu-int4-sym-cw
```

Açıkça verilen profil her zaman otomatik seçimin önüne geçer; plan bu durumda, profil verilmeseydi
hangisinin seçileceğini de belirtir. Bilinmeyen bir profil kimliği reddedilir; o aygıtın profilleri
ve `yonga registry profiles` komutu gösterilir. Grup boyutu modeli nicemleyemeyen (`config.json`
dosyasından anlaşılabildiği kadarıyla) açıkça verilmiş bir profil indirmeden önce reddedilir; yonga
onun yerine başka bir profil koymaz.

`cpu-int8-asym-ref` ve `cpu-fp16-ref` boyut aralığına bağlı varsayılanlar değil, referans
profilleridir; yonga `cpu-int8-asym-ref` profilini yalnızca hiçbir grup tabanlı CPU profili modele
uymadığında kendiliğinden seçer. `yonga eval` ile nicemleme kalitesini ölçmek için neredeyse kayıpsız
bir referans hazırlamaya yararlar; bkz.
[Sohbet, yoklama, ölçüm ve değerlendirme](run.md). Varsayılanların dayandığı ölçümler
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md#105-what-the-registry-does-with-it)
belgesindedir.

## Onaylar ve `--yes` { #confirmations }

yonga, beklemeyebileceğiniz bir şey yapmadan önce sorar. Plan bu soruları **Confirmation required**
altında listeler. Sorular şu durumları kapsar:

- yönetilen bir dışa aktarma ortamı kurmak;
- adını verdiğiniz depo yerine üçüncü taraf bir türevi dönüştürmek;
- 5 GiB'tan büyük bir indirme;
- dışa aktarılan modele bilinen bir onarımı uygulamak;
- var olan bir artifact dizininin yerine yenisini koymak (örneğin `--force` ile);
- `--device auto` kipinde, başarısız bir yoklamadan sonra başka bir aygıta geçmek (bu an geldiğinde
  sorulur);
- uzak kod çalıştırmak (bkz. [aşağısı](#remote-code)).

`--yes` (`-y`) sıradan sorulara evet yanıtı verir:

```bash
yonga prepare Qwen/Qwen3-0.6B --device cpu --yes
```

**Terminal yoksa** (bir betikte ya da CI işinde olduğu gibi stdin bir TTY değilse) ve `--yes`
verilmemişse soru sorulamaz; `prepare` 60 çıkış koduyla durur ve komutunuzu `--yes` eklenmiş hâliyle
yeniden yazdırır. Gerçek çıktı, kısaltılmış:

```text
✗ Preparation needs a confirmation

Confirmation 'repair' could not be asked, because this session has no terminal to ask in (stdin is
not a terminal): This export is expected to need a known artifact repair:
ov-tokenizer-precision-string-casing

  - Re-run the same command with --yes to approve it without a prompt: yonga … --offline prepare
Qwen/Qwen3-0.6B --device cpu --yes
  - Or run it in a terminal to be asked.
```

Bir soruya "hayır" yanıtı vermek de işi 60 çıkış koduyla iptal eder.

!!! warning "`--yes` neyi asla yapmaz"
    `--yes` uzak kodu asla etkinleştirmez, uzak kod sorusunu asla yanıtlamaz ve
    `--ignore-exporter-requirements` anlamına asla gelmez. Bunlar bilinçli ve ayrı kararlardır.

## Sürümler (revision) { #revisions }

yonga varsayılan olarak deponun varsayılan dalını hazırlar ve her şeyden önce onu bir commit'e
sabitler. Belirli bir dalı, etiketi ya da commit'i hazırlamak için iki biçimden birini kullanın:

```bash
yonga prepare Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
yonga prepare Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
```

İkisini birden verirseniz `--revision` geçerli olur. Artifact dizini, sabitlenen commit'in ilk 12
karakteri ve profilin adıyla adlandırılır (`c1899de289a0--npu-int4-sym-g128`); böylece aynı modelin
farklı sürümleri yan yana durur.

## Erişimi kısıtlı ve özel modeller { #gated }

Bazı depolar, sahibinin koşullarını kabul etmenizi ve oturum açmanızı ister. Üst verileri herkese açık
olduğu için `yonga inspect` genellikle yine çalışır; erişim belirteci gerektiren indirmedir. Belirteç
yoksa `prepare` hiçbir şey indirmeden reddeder (çıkış kodu 21).

Erişim almak için:

1. Hugging Face'teki model sayfasını tarayıcıda açın ve lisans ya da erişim koşullarını kabul edin.
2. Hugging Face'in `hf` komut satırı aracıyla oturum açın ya da bir erişim belirtecini
   `HF_TOKEN` ortam değişkeni olarak tanımlayın (`export`):

    ```bash
    hf auth login
    ```

3. Aynı `yonga prepare` komutunu yeniden çalıştırın.

`yonga doctor` son bölümünde bir kimlik bilgisinin kullanılıp kullanılmadığını gösterir. yonga bir
erişim belirtecini asla yazdırmaz ve iş günlükleri bir gizleyiciden geçirilir. Ayrıntılar:
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#55-gated-or-private-repositories).

## Uzak kod { #remote-code }

Bazı depolar ancak yayımladıkları Python kodu çalıştırılarak dönüştürülebilir. Siz açıkça
istemedikçe yonga bunu yapmaz. Bayrak verilmemişse `prepare` hiçbir iş yapmadan 22 çıkış koduyla
durur ve hangi bayrağı eklemeniz gerektiğini söyler.

İzin vermek için önce deponun özel kodunu gözden geçirin, sonra şunu çalıştırın:

```bash
yonga prepare MODEL --trust-remote-code
```

yonga yine de terminalde onay ister. Bu soru güvenlik açısından hassastır: `--yes` onu asla
yanıtlamaz ve terminali olmayan bir oturumda reddedilir (çıkış kodu 60). Uzak kod etkinken yalnızca
dönüştürmenin okuduğu dosyalar değil, deponun tamamı indirilir.

## GGUF depoları { #gguf }

OpenVINO GenAI bazı GGUF dosyalarını doğrudan içe aktarabilir. Yalnızca GGUF içeren bir depoda yonga
önce her aday dosyanın başlığını okur (tam indirme yapmaz), tensör türlerini içe aktarıcının
okuyabildiği ilk tercih edilen nicemlemeyi seçer ve yalnızca o dosyayı indirir. Plan dosyayı
**GGUF file** altında gösterir; atladığı dosyaları ve nedenlerini listeler.

```bash
yonga inspect Qwen/Qwen2.5-0.5B-Instruct-GGUF
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --yes
```

Dosyayı kendiniz seçmek için `--gguf-file` ile adını verin:

```bash
yonga prepare Qwen/Qwen2.5-0.5B-Instruct-GGUF --device gpu --gguf-file qwen2.5-0.5b-instruct-q8_0.gguf --yes
```

GGUF içe aktarma dosyanın kendi nicemlemesini korur, bu yüzden profiller uygulanmaz; artifact,
sürüm, dosyanın nicemlemesi ve aygıtla adlandırılır (örneğin `<revision>--gguf-q4_0-gpu`). `chat`, `bench`, `serve` ve
`eval` için normal bir artifact'tir, ancak NPU'da çalışmaz. yonga 0.5.0'ın yerleşik kayıt defteriyle
GGUF içe aktarma `llama`, `qwen2` ve `qwen3` mimarilerini kapsar. Bir GGUF deposu içe aktarılamıyorsa
`--source` ile aynı modelin bir safetensors deposunu gösterebilirsiniz; bkz.
[TROUBLESHOOTING.md](https://github.com/ooguz/yonga/blob/main/docs/TROUBLESHOOTING.md#58-a-gguf-repository-cannot-be-imported).

## Başka bir depoyu dönüştürmek { #source }

`--source REPO`, adını verdiğiniz depo yerine başka bir depoyu dönüştürür; örneğin kendi deposu
dönüştürülemeyen bir modelin safetensors kopyasını. Plandaki **Resolved source** ve **Relationship**
satırları neyin kullanıldığını gösterir ve artifact'in manifest dosyası kökeni kaydeder. Kayıt
defterinin kendisi bir model için üçüncü taraf bir türev bildiriyorsa yonga bunu planda önerebilir ve
dönüştürmeden önce sorar.

## Çevrimdışı çalışma { #offline }

Model zaten Hugging Face önbelleğinizdeyse ağ erişimi olmadan hazırlayabilirsiniz. Çevrimdışı kip,
üst veriler için bile Hub'a asla bağlanmaz. Şu yollardan biriyle açın:

=== "Seçenek"

    ```bash
    yonga --offline prepare Qwen/Qwen3-0.6B --device cpu
    ```

    `--offline` genel bir seçenektir: komut adından **önce** yazılır.

=== "yonga değişkeni"

    ```bash
    YONGA_OFFLINE=1 yonga prepare Qwen/Qwen3-0.6B --device cpu
    ```

=== "Hugging Face değişkeni"

    ```bash
    HF_HUB_OFFLINE=1 yonga prepare Qwen/Qwen3-0.6B --device cpu
    ```

Çevrimdışı kipin gerektirdikleri:

- Sürüm ve ağırlıkları yerel Hugging Face önbelleğinde olmalıdır. Sürüm belirtmezseniz yonga
  önbelleğin kaydettiği sürümü ya da önbellekteki tek anlık görüntüyü (snapshot) kullanır. Kaydı
  olmayan birden çok anlık görüntü varsa tahmin yürütmek yerine reddeder: birini `MODEL@<sha>`
  olarak belirtin.
- Dışa aktarma yönetilen bir ortam gerektiriyorsa ortam zaten var olmalıdır. Yeni bir ortam kurmak
  ağ gerektirir; bu yüzden yeni ortam gereken çevrimdışı bir iş durur (çıkış kodu 20). Var olan bir
  ortam yeniden kullanılır.

Önbellekte olmayan bir model 20 çıkış koduyla durur ve önbelleği nasıl dolduracağınızı söyler (gerçek
çıktı, kısaltılmış):

```text
✗ Not available offline

Offline mode is active and openai-community/gpt2-xl is not in the local Hugging Face cache
(…/.cache/huggingface/hub), so it cannot be used without network access.

  - Re-run without offline mode (no --offline, and YONGA_OFFLINE / HF_HUB_OFFLINE unset) to populate
the cache: …
  - Or point HF_HOME at a cache that already contains this revision.
```

## Yönetilen dışa aktarma ortamları { #managed-environments }

Bir modeli dönüştürmek, o mimarinin `transformers` içindeki modelleme kodunu çalıştırır ve bazı
mimariler belirli bir sürüm ister. yonga'nın yanındaki paketler bir modelin dışa aktarıcı
gereksinimlerini karşılayamadığında ya da hiç dışa aktarma araç zinciri olmadığında (AppImage ya da
`export` eki olmadan yapılmış bir pip kurulumu), dışa aktarma başarısız olmak yerine yonga'nın onun
için kurduğu yalıtılmış bir Python ortamında çalışır.

- Ortam ön denetim (preflight) sırasında, model indirilmeden **önce** kurulur; böylece başarısız bir
  kurulum birkaç gigabaytlık bir indirmeye değil, birkaç dakikaya mal olur.
- Önbellek dizininin altında durur (varsayılan olarak `~/.cache/yonga/envs/`), yaklaşık 3,0 GiB'a
  kadar yer kaplar ve yalnızca dışa aktarma tarafından kullanılır. Kendi Python ortamınız
  değiştirilmez.
- İlk kullanımdan önce doğrulanır ve aynı gereksinimlere sahip sonraki işlerde yeniden kullanılır.
- Ortam kurmak plandaki onaylardan biridir; `--yes` bunu onaylar.

Ortamları yönetmek için:

```bash
yonga env list          # ortamlar ve kapladıkları disk alanı
yonga env remove KEY    # birini siler; sonraki bir dışa aktarma gerektirirse yeniden kurulur
```

İki bayrak bu davranışı değiştirir:

| Bayrak | Etkisi |
|---|---|
| `--no-managed-env` | Asla ortam kurmaz. Kurulu paketler gereksinimleri karşılayamıyorsa indirmeden önce, ön denetimde durur (çıkış kodu 10). |
| `--ignore-exporter-requirements` | Bildirilmiş bir gereksinim karşılanmasa ya da kayıt defteri bu dışa aktarmanın kurulu sürümlerle başarısız olduğunu bilse bile kendi ortamınızda dışa aktarır. Bu geçersiz kılma manifest dosyasına kaydedilir. `--yes` bunu asla etkinleştirmez. |

Ayrıntılar: [INSTALL.md, bölüm 6](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#6-the-transformers-version-caveat).

## Disk alanı ve yönetilen geçici dizin { #disk-space }

Bir dönüştürme, modelin boyutunun çok ötesinde çalışma alanı ister. yonga asla `/tmp` içinde
dönüştürme yapmaz: önbellek dizininin altında yönetilen, diske dayalı bir dizin kullanır (varsayılan
olarak `~/.cache/yonga/tmp/<job-id>`) ve her dönüştürme süreci için `TMPDIR`, `TMP` ve `TEMP`
değişkenlerini oraya yönlendirir. Bu yüzden küçük ya da RAM'e dayalı bir `/tmp` hata değil, yalnızca
bir nottur. yonga için onu büyütmeyin.

Plan üç boyut gösterir:

| Satır | Nasıl tahmin edilir | Nereye gider |
|---|---|---|
| Expected download | Dönüştürmenin okuduğu dosyalar | Hugging Face önbelleği (`HF_HOME`) |
| Estimated temp requirement | 20 GiB ile kaynak ağırlıkların 2,5 katından büyük olanı, artı 5 GiB güvenlik payı | Önbellek dizini |
| Estimated artifact | Profilden ve parametre sayısından | Veri dizini (varsayılan olarak `~/.local/share/yonga`) |

Önbellek dizininin ya da artifact'in yazılacağı veri dizininin bulunduğu dosya sisteminde yer yoksa
ön denetim indirmeden önce 30 çıkış koduyla durur. Daha büyük bir disk kullanmak için önbellek
dizinini (isterseniz veri dizinini de) taşıyın:

```bash
export YONGA_CACHE_DIR=/mnt/big/yonga-cache
export YONGA_DATA_DIR=/mnt/big/yonga-data
```

ya da komut adından önce `--cache-dir` ve `--data-dir` verin. Kaynak model normal Hugging Face
önbelleğinde kalır ve orada asla değiştirilmez. Geçici dizin, iş başarıyla bitince silinir; bir
hatadan sonra ise tanılama için, sonraki bir temizlik onu kaldırana kadar saklanır (bkz. [İptal](#cancel)). Arka plan:
[INSTALL.md, bölüm 5](https://github.com/ooguz/yonga/blob/main/docs/INSTALL.md#5-tmp-tmpfs-and-why-yonga-does-not-ask-you-to-enlarge-it).

## Yeniden çalıştırma ve yeniden kullanma { #reuse }

Hazır bir artifact'i olan bir sürüm ve profil için `prepare` komutunu yeniden çalıştırmak o artifact'i
yeniden kullanır: indirme, dönüştürme ve soru yoktur; bu yüzden bir betikte `--yes` olmadan güvenle
çalışır. Sonuçta `Reused    an existing artifact at this revision and profile` yazar.

- **Başka bir aygıt.** Yeniden kullanım aygıta göre değil, sürüm ve profile göre belirlenir.
  Artifact'in hiç yoklanmadığı bir `--device` verirseniz yonga onu önce orada derler, kısa bir
  üretim denemesi yapar ve sonucu kaydeder. Orada çalışmazsa iş aygıt değiştirmeden durur ve
  artifact zaten çalıştığı aygıtlar için hazır kalır.
- **Baştan başlamak.** `--force` yeniden hazırlar ve var olan dizini değiştirmeden önce sorar.
- **Bir hatadan sonra yeniden denemek.** İndirmeler Hugging Face önbelleğinden kaldığı yerden devam
  eder. Aynı girdilere sahip önceki bir iş dışa aktarmayı bitirip daha sonraki bir aşamada başarısız
  olduysa (ya da iptal edildiyse), yeniden deneme o dışa aktarmanın her dosyasını denetler ve onu
  benimser: plan önceki işi adlandıran bir `Export` satırı gösterir ve indirme ile dönüştürme
  aşamaları atlanır. `--fresh-export` her durumda yeniden dışa aktarır. Kendisi yarıda kesilmiş bir
  dışa aktarma her zaman baştan başlar.
- **Değişen bir varsayılan.** Daha yeni bir yonga farklı bir varsayılan profil seçerse yeni artifact
  eskisinin üzerine değil, yanına oluşturulur ve plan eskisini yeniden kullanacak `--profile`
  değerini belirtir.

## Ctrl-C ile iptal etmek { #cancel }

Bir hazırlamayı durdurmak için istediğiniz an ++ctrl+c++ tuşlarına basın. İş 60 çıkış koduyla biter
ve `yonga logs` içinde `cancelled` olarak kaydedilir.

Neyin korunduğu, neyin korunmadığı:

- **Var olan artifact'lerinize asla dokunulmaz.** Yeni artifact gizli bir ara dizinde (staging)
  oluşturulur ve ancak sonunda atomik bir yeniden adlandırmayla artifact olur; bu yüzden iptal edilen
  bir iş bir artifact'i bozamaz ve `yonga list` yarım kalmış bir sonucu asla göstermez.
- **İndirilen dosyalar** Hugging Face önbelleğinde kalır; bir sonraki çalıştırma indirmeyi kaldığı
  yerden sürdürür.
- **Bitmiş bir dışa aktarma** ara dizinde kalır; böylece aynı girdilerle yapılan bir yeniden
  deneme onu kullanabilir (bkz. [yukarısı](#reuse)).
- İptal edilen bir işin **yönetilen geçici dizini** hemen silinir.
- **Yarıda kesilen bir ortam kurulumu** silinir; yarım hâliyle asla kullanılmaz.

Başarısız ya da iptal edilmiş işlerden artakalan ara dizinler ve geçici dizinler, sonraki bir iş
başladığında kendiliğinden kaldırılır (varsayılan olarak 72 saat sonra ve en fazla 3 tanesi
tutularak). `yonga list --clean` aynı temizliği istediğiniz an
çalıştırır.

## Bir şey ters gittiğinde: iş günlüğü { #logs }

Her `prepare` çalıştırması bir iş kimliği (planın alt kenarında gösterilir) ve veri dizini altında bir
günlük dosyası alır; varsayılan olarak `~/.local/share/yonga/logs/<job-id>.jsonl`. Hata mesajı bu
dosyanın yoluyla biter. Okumak için:

```bash
yonga logs                    # son işler, en yenisi önce
yonga logs JOB_ID             # tek bir iş, tablo olarak; benzersiz bir önek yeterli
yonga logs JOB_ID --raw       # JSON Lines dosyası, yazıldığı gibi
yonga logs JOB_ID --bundle    # hata raporuna eklenecek, gizlenmiş bir .tar.gz
```

```text
job       started           command  model                     device  status        took      log
──────────────────────────────────────────────────────────────────────────────────────────────────
c1ec9272  2026-10-11 20:09  prepare  Qwen/Qwen3-0.6B           CPU     ! cancelled  2.8 s  1.8 KiB
f57e7bd4  2026-10-11 20:09  prepare  openai-community/gpt2-xl  CPU     ✗ failed     <1 ms  1.1 KiB
```

Terminal kısa bir açıklama gösterse bile günlük ham hata metnini saklar. Bu metni terminalde de görmek
için genel `--debug` seçeneğini ekleyin (`yonga --debug prepare …`). Her tanılamanın ne anlama
geldiği ve nasıl düzeltileceği için bkz. [Sorun giderme](../troubleshooting.md).

## Tüm `prepare` seçenekleri { #options }

| Seçenek | Ne yapar |
|---|---|
| `MODEL` | Hugging Face depo kimliği; isteğe bağlı olarak `MODEL@REVISION`. |
| `--revision REV` | Dal, etiket ya da commit sha'sı. |
| `--device auto\|npu\|gpu\|cpu` | Hedef aygıt. Varsayılan `auto`. |
| `--profile ID` | Dönüştürme profili kimliği (`yonga registry profiles`). |
| `--source REPO` | İstenen depo yerine bu depoyu dönüştürür. |
| `--gguf-file FILE` | Bir GGUF deposu için içe aktarılacak dosya. |
| `--trust-remote-code` | Dönüştürmenin depoda yayımlanan Python kodunu çalıştırmasına izin verir. |
| `--ignore-exporter-requirements` | Bir dışa aktarıcı gereksinimi karşılanmasa bile bu ortamda dışa aktarır. |
| `--no-managed-env` | Asla yönetilen dışa aktarma ortamı kurmaz; bunun yerine reddeder. |
| `--fresh-export` | Başarısız bir işten kalan tamamlanmış dışa aktarmayı kullanmak yerine yeniden dışa aktarır. |
| `--yes`, `-y` | Sıradan sorulara evet yanıtı verir. |
| `--force` | Artifact var olsa bile yeniden hazırlar. |
| `--no-repair` | Dışa aktarılan modeli asla değiştirmez. Bir onarım gerekiyorsa iş durur (çıkış kodu 42). |
| `--json` | Sonucu tek bir JSON belgesi olarak yazdırır. |

Tam belirtim [CLI_SPEC.md, bölüm 6](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#6-prepare)
içindedir; tüm komutların özeti [Komut başvurusu](../reference/cli.md) sayfasındadır.
