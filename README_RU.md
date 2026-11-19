# DiPA: запуск

Реализация целевой функции статьи в виде Python-скриптов. Подробности — в [README](README.md), различия с исходным ноутбуком — в [описании метода](docs/method.md).

В ноутбуке был Sibling-Attack с IR-152 и моделью атрибутов. Здесь используется минимизация суммы cosine similarity с исходным лицом по ансамблю ArcFace, CosFace и MagFace, median pooling патча и опциональный TV.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Положите веса в `weights/` по [инструкции](weights/README.md). Веса в архив не входят. Для GPU установите подходящую CUDA-сборку PyTorch.

## Запуск

Запускайте команды из корня репозитория. Вход — заранее выровненное лицо. Автоматического детектора в скрипте нет; при наличии пяти landmarks можно использовать `prepare_face.py`.

```bash
python generate.py --image data/face.png --config configs/dipa.json --output outputs/dipa --device cuda --num-patches 5
python generate.py --image data/face.png --config configs/dipa_tv.json --output outputs/dipa_tv --device cuda --num-patches 5
```

На смартфоне показывайте `patch.png`. `latent_patch.png` — параметр до median pooling, его показывать не нужно. Для каждого патча сохраняются изображение лица с патчем, конфиг, seed, история оптимизации и метрики для сохранённых PNG.

Настройте положение патча через `attack.box`: `[x, y, ширина, высота]`. По умолчанию это `[45,160,160,90]` на лице 256 × 256. Размеры `canvas_size` и `screen_size` задаются как `[высота, ширина]`. Соотношение сторон `screen_size` должно совпадать с прямоугольником патча.

```bash
python evaluate.py --clean outputs/dipa/clean.png --patched outputs/dipa/patch_000/patched.png --config configs/dipa.json --output outputs/scores.json --device cuda
python -m unittest discover -s tests -v
```

Для обработки измерений камеры заполните CSV по заголовку `examples/camera_trials.csv` и запустите `summarize_camera.py`. Он отдельно считает смену идентичности и успешный dodging при сохранённой детекции.

В статье не указаны конкретные checkpoints, глубины сетей, параметры median pooling, optimizer и коэффициент TV. Значения в конфигах — явно выбранные настройки реализации, а не восстановленные параметры экспериментов статьи. Полное воспроизведение таблицы требует исходных весов, настроек и измерений камеры.
