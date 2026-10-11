# Test edilen donanım ve modeller

yonga belgelerinin aygıtlar, modeller ve hız hakkında söylediği her şey tek bir makinede ölçüldü.
Bu sayfa o makineyi, üzerinde doğrulanan model ailelerini, profil ölçümlerinden öne çıkanları ve
çalışmadığı bilinen şeyleri listeler. Bunu başka donanımlar için bir söz olarak değil, neyin
çalıştırıldığının kaydı olarak okuyun: başka hiçbir CPU, GPU, NPU, dağıtım ya da OpenVINO sürümü
test edilmedi.

## Referans makine { #reference-machine }

| | |
|---|---|
| CPU / platform | Intel Core Ultra 7 268V (Lunar Lake) |
| GPU | Intel Arc iGPU (`Intel(R) Arc(TM) Graphics (iGPU)`, `GPU: vendor=0x8086 arch=v20.4.4`) |
| NPU | Intel AI Boost, `/dev/accel/accel0`, `intel_vpu` çekirdek modülü, compiler-in-plugin |
| İşletim sistemi | Ubuntu 26.04.1 LTS, çekirdek 7.2.6 |
| Python | 3.14.4 (test takımı 3.11'de de geçiyor) |
| OpenVINO | 2026.4.0 |
| OpenVINO GenAI | 2026.4.0.0 |
| OpenVINO Tokenizers | 2026.4.0.0 |
| optimum-intel | 2.2.0 |
| nncf | 3.4.0 |
| transformers | Makinede 5.5.4; Qwen3.5 dışa aktarmaları, uyumluluk kayıt defterinin (registry) bu mimari için sabitlediği 5.2.0 sürümüyle yapıldı |
| huggingface_hub | 1.21.0 |

Ubuntu 24.04 hedeflenen bir dağıtımdır, ancak henüz test edilmedi. Kendi makinenizi bununla
karşılaştırmak için `yonga doctor` çalıştırın; sürücüler ve izinler
[Kurulum](installation.md) sayfasında anlatılır.

## O makinede doğrulanan model aileleri { #families }

Her aileden küçük bir model hazırlandı, GPU'da ve NPU'da üretim duman testiyle yoklandı ve CPU'da
performans ölçümü yapıldı:

| Aile | Model | CPU | GPU | NPU |
|---|---|---|---|---|
| qwen3 | `Qwen/Qwen3-0.6B` | ✓ | ✓ | ✓ 67.3 tok/s |
| qwen2 | `Qwen/Qwen2.5-0.5B-Instruct` | ✓ 23.8 tok/s | ✓ | ✓ |
| llama | `unsloth/Llama-3.2-1B-Instruct` | ✓ 12.6 tok/s | ✓ | ✓ |
| gemma3 (metin) | `unsloth/gemma-3-1b-it` | ✓ 15.3 tok/s | ✓ | ✓ |
| gemma2 | `unsloth/gemma-2-2b-it` | ✓ 7.2 tok/s | ✓ | ✓ |
| olmo2 | `allenai/OLMo-2-0425-1B-Instruct` | ✓ 13.9 tok/s | ✓ | ✓ |
| granite | `ibm-granite/granite-3.3-2b-instruct` | ✓ 7.6 tok/s | ✓ | ✓ |
| smollm3 | `HuggingFaceTB/SmolLM3-3B` | ✓ 6.2 tok/s | ✓ | ✓ |
| lfm2 | `LiquidAI/LFM2-350M` | ✓ 21.8 tok/s | ✓ | ✓ |
| qwen2.5-vl | `Qwen/Qwen2.5-VL-3B-Instruct` | ✓ 6.2 tok/s | ✓ | ✗ derleyici grafı reddediyor |
| qwen3.5 | referans 9B VLM | ✓ | ✓ | ✗ derleyici grafı reddediyor |
| phi3 | `microsoft/Phi-4-mini-instruct` | ✓ hazırlandı ve değerlendirildi | — denenmedi | ✓ 21.9 tok/s (grup bazında) |

CPU değerleri, kısa istemde tek bir çalıştırmanın INT4 üretim hızıdır. "Referans 9B VLM", yalnızca
GGUF yayımlayan `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` deposudur ve bildirilmiş
safetensors türevi üzerinden hazırlandı. Gemma, LFM2 ve Qwen2.5-VL kurulu olandan daha eski bir
Transformers gerektirir; `prepare` bunları yönetilen dışa aktarma ortamlarında dışa aktardı.

### "tested" ne demek { #what-tested-means }

Kayıt defterinde bu ailelerin aygıt iddiaları, yukarıdaki çalıştırmalar sayesinde `tested` kanıt
düzeyini taşır: bu makinede, bu paket sürümleriyle model o aygıtta hazırlandı, derlendi ve metin
üretti. Bu bir kanıttır, garanti değil:

- Bu **bir destek matrisi değildir**. Aynı ailenin başka bir modeli, başka bir boyutu, başka bir
  Intel nesli ya da başka bir OpenVINO sürümü farklı davranabilir.
- Tabloda olmayan bir aile de çalışabilir. Bu durumda `inferred` ya da `unknown` olarak
  değerlendirilir ve kararı `prepare`'in sonundaki yoklama verir. Kanıt düzeyleri
  [yonga nasıl çalışır](concepts.md#registry) sayfasında açıklanır.
- tok/s değerleri, küçük modellerin tek çalıştırmalarıdır ve örnek olarak verilmiştir. Bir
  performans iddiası değildir.

Önemsediğiniz bir model için `yonga inspect MODEL`, kayıt defterinin sizin makinenizde onun
hakkında bildiklerini gösterir.

### Gerçek bir NPU çalıştırması { #npu-run }

Referans makinede yonga 0.5.0 ile, `npu-int4-sym-g128` profiliyle yapılmış bir `Qwen/Qwen3-0.6B`
performans ölçümü (kısaltılmış):

```text
Model           Qwen/Qwen3-0.6B
Artifact        cc24636a2f49356a
Device          NPU
Profile         npu-int4-sym-g128
Suite           standard  (3 runs, 1 warmup)
Max new tokens  256
Compile         858 ms

Prompt               In tok   Out tok       TTFT   tok/s      TPOT    Wall
──────────────────────────────────────────────────────────────────────────
short-instruction        31       256   164.4 ms    68.2   14.7 ms   3.9 s
medium-explanation      104       256   164.7 ms    67.7   14.8 ms   3.9 s
long-context            623       256   165.0 ms    67.7   14.8 ms   3.9 s
```

Kendi ölçümünüzü nasıl yapacağınız [Sohbet, yoklama, ölçüm ve değerlendirme](guide/run.md)
sayfasında anlatılır.

## Profil ölçümleri { #measurements }

[Profiller](concepts.md#profiles) bölümünde anlatılan varsayılanlar, referans makinede `yonga eval`
ve `yonga bench` ile yapılan ölçümlere dayanır. `eval`, nicemleme (quantization) uygulanmış bir hazırlanmış modelin
(artifact) sonraki token tahminlerinin aynı ağırlıkların daha yüksek hassasiyetli bir referansından ne kadar
uzaklaştığını ölçer (ortalama KL ıraksaması, top-1 uyumu ve perplexity değişimi). Daha düşük
ıraksama, nicemlemenin daha az zarar verdiği anlamına gelir; gerçek görevlerde yanıt kalitesinin
bir ölçüsü değildir. Tüm tablolar ve çekinceler
[PROFILE_COMPARISON.md](https://github.com/ooguz/yonga/blob/main/docs/PROFILE_COMPARISON.md)
belgesindedir.

### 5B'nin altında katmanların beşte birini INT8'de tutmak { #ratio-08 }

0,35B ile 8,19B parametre arasındaki on iki model, INT8 referansına karşı oran 1,0 ve oran 0,8 ile
INT4 olarak ölçüldü:

- Oran 0,8, **her modelde** referansa daha yakındır: KL'de %16-42 ve top-1 uyumunda 2-6 puan
  daha iyi, karşılığında **%9-18 daha fazla disk**.
- 5B'nin altında oran 1,0 perplexity'yi çoğu zaman %20-33 artırır; oran 0,8 bu artışın beşte biri
  ile üçte ikisi arasındaki kısmını giderir. 8B'de perplexity artışı oran 1,0 ile +%7,2, oran 0,8 ile +%6,8'dir.
- Hızda oran 0,8, 5B'nin altında GPU'nun saniyedeki token sayısının %10-18'ine, 8B'de %22'sine ve
  CPU'nunkinin %12'ye kadarına mal olur. Bunu "yaklaşık beşte bire kadar" diye okuyun: makinenin
  çalıştırmadan çalıştırmaya oynaması yaklaşık %20'ydi.

`-r08` profillerinin 5B parametrenin altında CPU ve GPU varsayılanı olmasının nedeni budur.

### NPU'nun iki INT4 biçimi { #npu-formats }

- Kanal bazında INT4 (`npu-int4-sym-cw`), ölçülen her modelde ve her boyutta, referanstan grup
  bazında INT4'e (`npu-int4-sym-g128`) göre **kabaca iki ila beş kat** daha fazla uzaklaşır. Grup
  bazındaki her artifact NPU'da derlendi ve metin üretti.
- Grup bazında biçim daha yavaştır: Qwen3'te 0,6B'de %14 yavaşlatır, 1,7B'de hiç yavaşlatmaz, 4B'de
  ise üçte bir yavaşlatır ve ilk token süresini iki katına çıkarır. Diğer ailelerde NPU'nun saniyedeki token sayısının
  onda biri ile üçte biri arasında bir kısmına mal olur.
- 8B'de kanal bazında biçim üçte bir daha hızlı ve çok daha az isabetlidir: perplexity artışı kanal
  bazında +%42,7, grup bazında +%9,4.

Bu yüzden NPU varsayılanı 5B'nin altında grup bazında, 5B'den itibaren kanal bazındadır. Küçük bir
modelde hızlı seçenek `--profile npu-int4-sym-cw`, büyük bir modelde kaliteli seçenek
`--profile npu-int4-sym-g128`'dir.

`yonga bench --suite standard` ile Qwen3 hızı; saniyedeki token sayısı, parantez içinde ilk token
süresi. GPU sütunları CPU profillerinin artifact'lerini kullanır; bu yüzden GPU profillerinin
`ATTENTION_BACKEND=SDPA` ayarı olmadan çalışırlar:

| Qwen3 | GPU r1.0 | GPU r0.8 | CPU r1.0 | CPU r0.8 | NPU cw | NPU g128 |
|---|---|---|---|---|---|---|
| 0.6B | 152.5 (27 ms) | 125.5 (49 ms) | 85.6 (41 ms) | 82.6 (41 ms) | 77.4 (123 ms) | 66.9 (168 ms) |
| 1.7B | 74.4 (47 ms) | 67.1 (56 ms) | 44.4 (101 ms) | 39.9 (109 ms) | 35.2 (319 ms) | 36.9 (431 ms) |
| 4B | 34.7 (60 ms) | 30.4 (74 ms) | 19.2 (276 ms) | 17.0 (292 ms) | 24.6 (832 ms) | 16.5 (1651 ms) |
| 8B | 16.6 (96 ms) | 13.0 (128 ms) | 10.0 (625 ms) | 9.7 (547 ms) | 13.9 (1785 ms) | 10.6 (2842 ms) |

### GPU'da 9B bir VLM { #vlm-9b }

Aynı Qwen3.5-9B ağırlıklarının iki INT4 nicemlemesi GPU'da karşılaştırıldı: `gpu-int4-asym-g128`
ve NPU'nun kanal bazındaki `npu-int4-sym-cw` profili.

- Kanal bazındaki artifact %2,17 daha küçüktür ve ikisi arasındaki hız farkı, çalıştırmadan
  çalıştırmaya oynamanın içinde kaldı.
- Grup bazındaki asimetrik artifact, INT8 referansına çok daha yakın kalır: ortalama KL 0,080
  (kanal bazında 0,197), top-1 uyumu %86,8 (%81,8), perplexity artışı +%7,1 (+%22,4). Bu fark sekiz
  metin parçasının her birinde, hem CPU'da hem GPU'da görüldü.
- İkisi de en çok Türkçe metinde, en az Python kodunda kaybeder; yani yalnızca İngilizce düzyazı
  üzerinde yapılan bir değerlendirme, nicemlemenin bedelini olduğundan küçük gösterir.

### GGUF içe aktarmaları { #gguf }

`Qwen/Qwen2.5-0.5B-Instruct` için içe aktarılan GGUF dosyalarının, nicemlenmemiş olsa bile yaklaşık
0,03 nat'lık bir ıraksama tabanı vardır; optimum ile yapılan INT8 dışa aktarması ise 0,006'ya
iner. Q8_0 ve Q6_K bu tabanda durur. GGUF Q4_0 dosyası referansa, yonga'nın oran 1,0'lık INT4
dışa aktarmasından daha yakındı (KL 0,199; yonga'nın dışa aktarmasında 0,290).

!!! note "Not"
    Bu sayıların hepsi tek bir dizüstü bilgisayardan, tek bir sürücüden, tek bir OpenVINO
    sürümünden ve tek bir ısıl durumdan gelir. Varsayılanların neden böyle olduğunu açıklarlar;
    sizin makinenizin ne ölçeceğini öngörmezler.

## Çalışmadığı bilinenler { #not-working }

### NPU'da görüntü-dil modelleri { #vlm-npu }

NPU derleyicisi, test edilen iki VLM ailesinin de graflarını reddeder: Qwen3.5 (9B referans model)
ve Qwen2.5-VL (`Qwen/Qwen2.5-VL-3B-Instruct`). Sürücü ve derleyici sağlıklıdır; bu NPU
derleyicisinin kabul edemediği şey grafın kendisidir. Kayıt defteri NPU'yu her ikisi için de test
edilmiş kanıtla desteklenmiyor olarak kaydeder; bu yüzden:

- `--device auto` bunlar için NPU'yu asla seçmez;
- açıkça verilen `--device npu` indirmeden önce, 50 çıkış koduyla reddedilir ve bunun yerine
  modeli GPU ya da CPU için hazırlayan komutlar gösterilir.

Bu modeller için GPU'yu ya da CPU'yu kullanın.

### Python 3.14'te phi3 ve mistral dışa aktarmaları { #python-314 }

optimum 2.3.0 ve öncesi, Python 3.14'te aralarında phi3 (Phi-4-mini) ve mistral'in de bulunduğu bazı
mimarilerin dışa aktarma yapılandırmasını oluşturamaz. Düzeltme optimum projesinde hazır (pull
request 2496), ancak henüz hiçbir optimum sürümü onu içermiyor. Python 3.11, 3.12 ve 3.13 bundan
etkilenmez.

yonga bunu bilir. Dışa aktarmanın Python 3.14'te ve etkilenen bir optimum ile çalışacağını
anlayabildiğinde indirmeden önce reddeder; anlayamadığında hatayı sınıflandırır. Her iki durumda
da kod `export.optimum.normalized_config`, çıkış kodu 10'dur. AppImage Python 3.14 içerir ve
yönetilen dışa aktarma ortamlarını onun üzerine kurar; bu yüzden bu modelleri dışa aktaramaz.
Bunun yerine yonga'yı pip ile Python 3.11-3.13 üzerinde bir sanal ortama kurun:

```bash
python3.13 -m venv ~/.venvs/yonga
~/.venvs/yonga/bin/pip install "yonga[openvino,export]"
```

Aile tablosundaki phi3 sonuçları da, PROFILE_COMPARISON.md'deki phi3 ve mistral satırları da
Python 3.14'te optimum 2.3.0'a bu henüz yayımlanmamış düzeltme eklenerek yapılan bir dışa
aktarmadan gelir. Yayımlanmış bir yonga optimum'u sizin için yamalayamaz.

### NPU'da Granite { #granite-npu }

`ibm-granite/granite-3.3-2b-instruct` NPU'da çalışır, ancak orada her iki INT4 biçiminde de
yavaştır: kendi boyutundaki modeller 25-40 tok/s'ye ulaşırken yaklaşık 4,5 tok/s. Nedeni
belirlenemedi.

### Windows { #windows }

Windows henüz desteklenmiyor. yonga yalnızca Linux x86_64 üzerinde çalışır: her aygıt, sürücü ve
yol denetimi Linux'a göre yazılmış ve Linux'ta test edilmiştir.
