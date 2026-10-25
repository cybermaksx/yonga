# OpenAI uyumlu sunucu

`yonga serve`, hazırlanmış bir modeli (artifact) OpenAI'ın chat completions biçimini konuşan yerel bir HTTP API'nin arkasına koyar; böylece OpenAI ile çalışan düzenleyiciler, betikler ve sohbet arayüzleri NPU'nuzu, GPU'nuzu veya CPU'nuzu kullanabilir. `yonga chat` ile aynı oturumu, aynı sohbet profili, üretim varsayılanları ve gizli akıl yürütmeyle çalıştırır. Bu sayfa sunucuyu nasıl başlatacağınızı, ona `curl` ve resmi `openai` Python istemcisiyle nasıl istek göndereceğinizi, neyi kabul edip neyi reddettiğini anlatır.

## Sunucuyu başlatmak {#start}

Hazırlanmış bir modeli sunmak için şunu çalıştırın:

```bash
yonga serve Qwen/Qwen3-0.6B
```

yonga modeli dinlemeye başlamadan önce yükler; böylece aygıtta yüklenemeyen bir model her isteği değil, `serve` komutunun kendisini başarısız kılar. Hazır olduğunda adresi yazdırır:

```text
Loading the model on NPU...
Serving Qwen/Qwen3-0.6B on NPU at http://127.0.0.1:8000/v1
Press Ctrl+C to stop.
```

Durdurmak için Ctrl+C'ye basın. Kullanımda olan bir port, model yüklenmeden önce bildirilir.

!!! note "Not"
    `serve`, `serve` ekini (FastAPI, uvicorn ve Pillow) gerektirir. AppImage bunu içerir; pip ile `yonga[serve]` kurun. Bkz. [Kurulum](../installation.md).

### Seçenekler {#options}

| Seçenek | Varsayılan | Ne yapar |
|---|---|---|
| `--host ADDRESS` | `127.0.0.1` | Dinlenecek adres. Siz aksini söylemedikçe yalnızca loopback. |
| `--port N` | `8000` | Dinlenecek TCP portu. |
| `--device auto\|npu\|gpu\|cpu` | `chat` ile aynı | Modelin çalışacağı aygıt. Verilmezse önce ayarlarınızdaki varsayılan aygıt, ardından artifact'in hazırlandığı aygıt kullanılır. |
| `--max-new-tokens N` | sohbet profilininki | İstek bir sınır belirtmediğinde çıktı token sınırı. |
| `--show-reasoning` | kapalı | Akıl yürütme bloklarını gizlemek yerine döndürür. |
| `--max-body-mb N` | `8` | MiB cinsinden en büyük istek gövdesi, 1 ile 64 arası. Gömülü görseller buna dahildir. |
| `--carry-images-forward` | kapalı | Önceki mesajlarla gönderilen görselleri reddetmek yerine son mesaja taşır (bkz. [Görseller](#images)). |

`serve`, `--json` kabul etmez; sunucunun durumunu bunun yerine `GET /health` JSON olarak bildirir. API anahtarı için bir seçenek yoktur: anahtar `YONGA_API_KEY` ortam değişkeninden okunur (bkz. [Başka makinelerden erişim](#api-key)).

## Uç noktalar {#endpoints}

| Yöntem ve yol | Ne döndürür |
|---|---|
| `GET /v1/models` | Sunulan tek modeli, OpenAI'ın model nesnesi biçiminde. |
| `GET /v1/models/{id}` | `{id}` onu adlandırıyorsa aynı modeli. |
| `POST /v1/chat/completions` | Akışlı veya akışsız OpenAI chat completions. |
| `GET /health` | `status`, `model`, `device`, `pipeline` ve `uptime_seconds`. Hiçbir zaman anahtar istemez. |

Model, `model` alanında üç ada yanıt verir: `yonga serve`'e verdiğiniz ad, artifact'in kimliği ve kaynak deposu. `yonga serve Qwen/Qwen3-0.6B` ile bir istek `Qwen/Qwen3-0.6B` adını ya da artifact kimliğini kullanabilir; `GET /v1/models`, `serve`'e verdiğiniz adı modelin `id` değeri olarak listeler. Başka bir ad 404 `model_not_found` alır.

## curl ile çağırmak {#curl}

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Say hello in one short sentence."}]}'
```

Referans makineden (NPU, Qwen3-0.6B) gerçek bir yanıt:

```text
{"id":"chatcmpl-…","object":"chat.completion","created":1791738528,"model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?"},
 "finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":10,"total_tokens":29}}
```

## openai Python istemcisini kullanmak {#python}

Temel URL'sini sunucuya yönelttiğinizde her OpenAI istemcisi çalışır. İstemci bir API anahtarı değeri ister; sunucuda anahtar ayarlı değilse herhangi bir metin yeterlidir.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

reply = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is an NPU? One sentence."}],
)
print(reply.choices[0].message.content)
```

Sohbetin sahibi istemcidir: her istekte sohbetin tamamını gönderin. Sunucu istekler arasında hiçbir şey saklamaz.

## Akış (streaming) {#streaming}

Yanıtı `data: [DONE]` ile biten, sunucu tarafından gönderilen olaylar (server-sent events) olarak almak için `stream` alanını ayarlayın:

=== "Python"

    ```python
    stream = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[{"role": "user", "content": "Name three rivers in Turkey."}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    ```

=== "curl"

    ```bash
    curl -sN http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Count from 1 to 5."}], "stream": true}'
    ```

`stream_options: {"include_usage": true}`, token kullanımını içeren son bir parça ekler. Gizli akıl yürütme akış sırasında da gizli kalır. Bağlantısı kopan bir istemcinin üretimi durdurulur.

## Üretim ayarları {#generation}

Bir istek `max_tokens` (veya `max_completion_tokens`), `temperature` (`0` açgözlü kod çözmedir), `top_p`, `top_k`, `seed` ve `repetition_penalty` belirtebilir. Bunlar [`yonga chat`](run.md#chat-commands) içindeki karşılık gelen eğik çizgi komutları gibi aralık denetiminden geçer (`temperature` 0 ile 2 arası, `top_p` 0'dan büyük ve en fazla 1, `top_k` 0 veya daha büyük); aralık dışındaki bir değer, alanı adlandıran bir 400 alır. yonga'nın tanımadığı alanlar yok sayılır.

### Durdurma dizileri {#stop}

`stop`, OpenAI'ın API'sinde olduğu gibi bir metin ya da her biri en fazla 256 karakterlik, en fazla 4 metinden oluşan bir liste alır:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "List three colours, one per line."}], "stop": ["\n"]}'
```

Referans makinede bu istek `"content":"1. Red"` ve `"finish_reason":"stop"` ile yanıtlandı. Yanıt, ilk durdurma dizisinden önce ve onu içermeden biter; üretim de orada durur, akışlı olsun olmasın. yonga durdurma dizilerini yalnızca görünen yanıtta arar; böylece `"\n\n"` gibi bir dizi gizli bir akıl yürütme bloğunun içinde tetiklenip size boş bir yanıt bırakamaz. `--show-reasoning` ile akıl yürütme görünen yanıtın parçası olduğundan, bir durdurma dizisi onu orada bitirebilir.

## Araç çağırma {#tools}

Araçlar, kayıt defterinin referans makinede kullanılabilir araç çağrıları yazdığını gördüğü model ailelerinde çalışır: **Qwen2.5** (Instruct modelleri), **Qwen3** (MoE modelleri hariç), **SmolLM3** ve **Granite**. Bir modelde bu özellik varsa `yonga registry explain MODEL` sohbet profilinin altında "Tool calling" satırını gösterir. Diğer modeller `tools` alanını tahmin yürütmek yerine, nedenini söyleyen bir 400 `serve.request.tools_unsupported` ile reddeder.

OpenAI `tools` alanını gönderin; yanıt `message.tool_calls` olarak, `finish_reason` değeri `tool_calls` ile gelir. Sonucu, tam OpenAI'daki gibi bir `tool` mesajıyla geri gönderin. Bu örnek NPU'da Qwen3-0.6B'ye karşı çalıştırıldı:

```python
import json

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What is the weather in Ankara? Use the tool."}]

reply = client.chat.completions.create(model="Qwen/Qwen3-0.6B", messages=messages, tools=tools)
message = reply.choices[0].message
if not message.tool_calls:
    print(message.content)  # the model answered without calling a tool
else:
    messages.append(message)
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        result = {"city": args.get("city"), "celsius": 21}  # run your real function here
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B", messages=messages, tools=tools
    )
    print(final.choices[0].message.content)
```

- `tool_choice`, `auto` (varsayılan) veya `none` olabilir. `required` ya da adı verilmiş bir fonksiyon 400 `unsupported_parameter` ile reddedilir: bir çağrıyı zorlamak kısıtlı kod çözme (constrained decoding) gerektirirdi.
- `parallel_tool_calls: false`, birden fazla çağrı yapan bir yanıtın yalnızca ilk çağrısını tutar.
- Akışta metin, bir çağrının başlayabileceği noktaya kadar gelir; ardından her çağrı bütün olarak gelir, bu yüzden model çağrıyı yazarken akış duraklar.
- Çağrıya benzeyen ama geçerli olmayan bir yanıt (bozuk JSON, sunmadığınız bir fonksiyon) tahmin edilmiş bir çağrı olarak değil, düz metin olarak döner. Modelin yalnızca akıl yürütürken taslağını çizdiği bir çağrı asla yapılmaz.
- İstek başına en fazla 128 araç.

!!! tip "İpucu"
    Küçük modeller araçları güvenilir biçimde çağırmaz; bu yüzden örnekteki gibi `tool_calls` içermeyen bir yanıtı her zaman ele alın. Araç şemaları her istemi de uzatır; bu NPU'da önemlidir (bkz. [NPU istem uzunluğu](#npu-prompt-length)).

## Görseller {#images}

Bir VLM artifact'i (Qwen2.5-VL, Qwen3.5) görsel de alır: `data:` URL'si taşıyan OpenAI `image_url` içerik parçaları olarak. Bir LLM artifact'i görselleri `serve.request.images_unsupported` ile reddeder.

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<EOF
{"model": "ARTIFACT", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG"}}]}]}
EOF
```

Kurallar:

- **Yalnızca `data:` URL'leri**, base64 ile kodlanmış PNG, JPEG, WebP veya GIF. yonga hiçbir görseli kendisi indirmez; bu yüzden `http(s)://` ve `file://` URL'leri 400 ile reddedilir ve bunlara izin veren bir seçenek yoktur: istemcilerinin adlandırdığı şeyi indiren bir sunucu, bulunduğu makinedeki ya da ağdaki herhangi bir yere erişmeye zorlanabilir.
- **İstek başına en fazla 4 görsel**, tüm mesajlar üzerinden sayılır; fazlası 400 `serve.request.too_many_images` alır.
- **Görsel başına en fazla 4096 × 4096 piksel (genişlik çarpı yükseklik, yaklaşık 16,8 milyon).** Daha büyük bir görsel küçültülmez, reddedilir; göndermeden önce küçültün.
- **Gövde boyutu.** Görseller 8 MiB'lık istek sınırına dahildir ve base64 boyutlarına üçte bir ekler. Daha büyük fotoğraflar için sınırı `--max-body-mb` ile (en fazla 64) yükseltin.
- **Görseller yalnızca son kullanıcı mesajıyla gider.** OpenVINO GenAI bir görseli daha önceki bir tura bağlayamaz; bu yüzden önceki bir mesajdaki görsel `serve.request.images_earlier_turn` ile reddedilir.

Open WebUI gibi sohbet arayüzleri, sohbetin önceki görsellerini her takip sorusunda yeniden gönderir. Bunlar için sunucuyu `--carry-images-forward` ile başlatın: önceki görseller, modele daha önce eklendiklerini söyleyen bir satırla birlikte en yeni mesaja taşınır. Model artık hangi görselin hangi sözlerle gönderildiğini bilmez; bu yüzden "gönderdiğim ilk resim" hakkındaki bir soru daha zayıf yanıtlanabilir. 4 görsel sınırı yine hepsini sayar.

## Başka makinelerden erişim ve API anahtarları {#api-key}

Sunucu varsayılan olarak yalnızca `127.0.0.1` adresini dinler; makinenizin dışından hiçbir şey ona ulaşamaz. Başka bir adrese bağlanmak (örneğin `--host 0.0.0.0`) bir uyarı yazdırır; anahtar ayarlı değilse uyarı bunu da belirtir. Anahtar ayarlıysa başlangıç satırı `(API key required)` ile biter.

Loopback dışında bir adrese bağladığınız her durumda ortamda `YONGA_API_KEY` ayarlayın:

```bash
export YONGA_API_KEY='choose-a-long-random-string'
yonga serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
```

Bu durumda `/health` dışındaki her uç nokta `Authorization: Bearer <key>` başlığını ister; başlığı olmayan bir istek 401 `invalid_api_key` alır. `openai` istemcisinde anahtarı `api_key` olarak verin. Anahtar bir seçenekten değil, yalnızca ortamdan okunur; böylece süreç listesinde görünmez ve hiçbir zaman günlüğe yazılmaz.

!!! warning "Uyarı"
    yonga düz HTTP sunar. Sunucuyu kendi makinenizin dışına açarsanız güvendiğiniz bir ağda tutun ya da TLS ekleyen bir ters vekil sunucunun (reverse proxy) arkasına koyun.

## OpenCode ile kullanın {#opencode}

```bash
yonga connect opencode 073d5d7c8ea75283 --port 8765
yonga serve 073d5d7c8ea75283 --port 8765
```

İlk komut, yapıtı OpenCode'un yapılandırmasına (`~/.config/opencode/opencode.json` ya da `--config` ile verdiğiniz dosya) bir sağlayıcı olarak ekler; bağlam ve çıktı sınırlarını yapıtın kendisinden okur. Değişikliği gösterir ve yazmadan önce sorar; `--yes` soruyu atlar. Ardından yazdırdığı komutla sunucuyu başlatın ve `opencode` içinde `yonga-8765/073d5d7c8ea75283` modelini seçin. Kayıt zaten doğruysa komutu yeniden çalıştırmak hiçbir şeyi değiştirmez.

OpenCode yapıtı kullanamayacaksa yonga reddeder ve hiçbir şey yazmaz. OpenCode her istekle 12 araç ve yaklaşık 5300 belirteçlik bir sistem istemi gönderir; bu yüzden:

- Araç çağırmayan bir yapıt reddedilir (`connect.opencode.tools_unsupported`).
- İstem uzunluğu 8192 belirtecin altında sabitlenmiş bir yapıt reddedilir (`connect.opencode.prompt_too_short`). Yerleşik NPU profillerinin hepsi 1024'te sabittir. Aynı modelin bir GPU ya da CPU yapıtını kullanın; ileti birini adıyla gösterir.
- Yorum ya da sondaki virgül içeren (JSONC) bir yapılandırma dosyası yeniden yazılmaz (`connect.opencode.config_not_json`). Yazdırdığı JSON'dan sağlayıcıyı elle ekleyin.

`YONGA_API_KEY` tanımlıysa kayıt ona `{env:YONGA_API_KEY}` olarak başvurur; anahtarın kendisi hiçbir zaman yazılmaz. Zaten var olan bir sağlayıcıya yonga yalnızca `npm`, `options.baseURL` ve `options.apiKey` alanlarını yazar; sağlayıcının adı ve sizin eklediğiniz diğer seçenekler olduğu gibi kalır.

```bash
yonga connect opencode --remove 073d5d7c8ea75283
```

`--remove` yalnızca yonga'nın yazdığı kayıtları siler; `yonga remove` yapıtı sildikten sonra da çalışır. Sonradan elle değiştirdiğiniz bir kayıt bırakılır ve listelenir, komut 70 koduyla çıkar. Yapılandırma dosyası hiçbir zaman silinmez. `--config FILE` ile yapılan bir bağlantı aynı `--config FILE` ile kaldırılır; verilmezse `--remove` bağlantının bulunduğu dosyayı adıyla gösterir, `yonga remove` da komutun tamamını yazdırır.

## Sınırlar ve reddedilen istekler {#limits}

yonga, yerine getiremeyeceği bir isteğe sessizce farklı bir yanıt vermek yerine onu reddeder. Hatalar OpenAI'ın `{"error": {"message", "type", "param", "code"}}` biçimini kullanır ve `param` sorunlu alanı adlandırır.

| İstek | Yanıt |
|---|---|
| 1 dışında bir `n` | 400 `unsupported_parameter` |
| sıfır olmayan `presence_penalty` veya `frequency_penalty` | 400 `unsupported_parameter` (`repetition_penalty` kullanın) |
| metin dışında bir `response_format` (JSON modu, JSON şeması) | 400 `unsupported_parameter` |
| `logprobs` | 400 `unsupported_parameter` |
| eski `functions` / `function_call` | 400 `unsupported_parameter` (`tools` kullanın) |
| `tool_choice` için `required` veya adı verilmiş bir fonksiyon | 400 `unsupported_parameter` |
| son mesaj ne kullanıcıdan ne de bir `tool` sonucu | 400 `unsupported_parameter` |
| kabul edilen bir `data:` URL'si olmayan görsel URL'si | 400 `unsupported_parameter` |
| 4096'dan fazla mesaj | 400 `invalid_request` |
| sunucunun sunmadığı bir `model` | 404 `model_not_found` |
| `--max-body-mb` değerinden (varsayılan 8 MiB) büyük gövde | 413 `request_too_large` |
| `YONGA_API_KEY` ayarlıyken eksik veya yanlış anahtar | 401 `invalid_api_key` |

Üretim sırasındaki bir çalışma zamanı hatası, `code` değeri yonga'nın tanılama kodu olan bir 500'dür. `stream: true` ile, üretim başlamadan reddedilen bir istek, akışsız haliyle aynı durumu ve gövdeyi alır; yalnızca akışın ortasındaki bir hata son bir `data: {"error": …}` olayı olarak gelir. Kodların tam listesi [serve belirtiminde](https://github.com/ooguz/yonga/blob/main/docs/CLI_SPEC.md#14-serve).

### NPU istem uzunluğu {#npu-prompt-length}

NPU'da istem uzunluğu, model derlenirken sabitlenir: yerleşik NPU profilleriyle 1024 token. Sohbetin tamamı, sunulan araçlar ve görseller dahil daha uzun bir istem, akışlı olsun olmasın 400 `context_length_exceeded` ile reddedilir. Referans makineden gerçek ileti:

```text
The prompt is too long for this model: It has 3013 tokens, and this model on NPU takes at most 1024:
its prompt length (MAX_PROMPT_LEN) is fixed when it is compiled. Shorten the conversation, or use a
GPU or CPU artifact of this model, whose prompt length is not fixed.
```

Sohbeti kısaltın, daha az araç sunun ya da aynı modelin GPU veya CPU artifact'ini sunun.

### Aynı anda tek istek {#one-at-a-time}

Sunucu tek bir aygıtta tek bir model tutar ve istekleri birer birer yanıtlar. Diğer istekler sıralarını bekler. Sunucu, yerel bir makinede tek bir kullanıcı ve onun araçları için tasarlanmıştır; aynı anda çok sayıda kullanıcı için değil.

## Sonraki adımlar {#next-steps}

- Aynı modeli etkileşimli kullanın: [Sohbet, yoklama, ölçüm ve değerlendirme](run.md).
- `serve` başlamıyorsa veya bir istek başarısız oluyorsa: [Sorun giderme](../troubleshooting.md).
- Her seçenek: [Komut başvurusu](../reference/cli.md).
