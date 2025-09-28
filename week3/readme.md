1) Файлы и структура

В корне: index.html, style.css, data.js, script.js, tf.min.js, u.data, u.item.

Подключения в <head> строго так:
style.css → tf.min.js → data.js → script.js.

2) Парсинг данных (data.js)

Считать ./u.item (pipe |) → собрать:

movies: [{rawId,index,title,genres[]}]

movieIndexByRawId: Map<rawItemId → i>

Считать ./u.data (TSV) → собрать:

userIndexByRawId: Map<rawUserId → u>

ratingsTriples: [{u,i,r}] (u/i — плотные индексы)

userRatedItems: Map<u, Set<i>>

STATS: {nUsers,nItems,nRatings,mean(μ)}

3) Модель MF (script.js)

Параметры по умолчанию (редактируемые в UI):
k=32, epochs=15, batch=2048, lr=0.01, lambda=1e-4.

Создать TF-переменные:

P:[U,k], Q:[I,k] ~ N(0,0.01), bu:[U]=0, bi:[I]=0, mu=μ.

Лосс: MSE по наблюдаемым + L2 на P,Q,bu,bi с lambda.

Оптимизатор: Adam(lr).

4) Обучение

Разбить ratingsTriples на train/val=90/10.

Учить батчами; после каждого батча: await tf.nextFrame(); все вычисления в tf.tidy(...).

Показывать в UI: epoch, Train RMSE, Val RMSE.

Кнопка Cancel прерывает цикл.

При повторном обучении — обязательно dispose() всех Variables/optimizer.

5) Инференс и рекомендации

Функция predictRating(u,i) = clip( μ + bu[u] + bi[i] + dot(P[u],Q[i]), 1,5 ).

recommendForUser(u, topN):

предсказать для всех i,

исключить userRatedItems.get(u),

отсортировать по предсказанию,

выдать Top-N c полями: Title, Genres, Predicted, Explain( μ | bᵤ | bᵢ | dot ).

6) UI (index.html + style.css)

Кнопки/поля:

#btn-load, статус #status, счётчики #stat-users/items/ratings/mu

#param-k #param-epochs #param-batch #param-lr #param-lambda

#btn-train #btn-cancel, прогресс #train-progress, метрики #train-epoch #train-trainrmse #train-valrmse

#user-select, #topn, #btn-recommend, таблица #results-table

Тёмная apple-like тема; контейнер max-width ~ 960px; доступные фокусы.

7) Производительность/надёжность

Batch ≤ 4096 (по умолчанию 2048).

Без утечек: все временные тензоры в tf.tidy; на перезапуск — disposeModel().

Исключить фризы: await tf.nextFrame() каждые 1–2 батча.

Читабельные ошибки парсинга/обучения; консоль чистая.

8) Acceptance Criteria

Данные загрузились; счётчики Users/Items/Ratings/μ корректны.

Обучение идёт; Train RMSE ↓, Val RMSE стабилизируется.

Рекомендации для пользователя с ≥5 оценками есть; уже оценённые исключены.

В «Explain» сходится: μ + bᵤ + bᵢ + dot ≈ Predicted (±1e-3).

Cancel работает; повторная тренировка без утечек/ошибок.

9) Запуск/деплой

Локально: python3 -m http.server → http://localhost:8000.

GitHub Pages: все файлы в корне/docs/; пути к данным — относительные (./u.data, ./u.item).
