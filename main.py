import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoostClassifier

df = pd.read_csv('datasets/csgoEDA.csv') 

st.set_page_config(
    page_title="ML Project Dashboard",
    layout="centered"
)

def show_about_page():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
            image = Image.open(r'images/photo.jpg') 
            st.image(image, caption='Разработчик ML-моделей', use_container_width=True)

    with col2:
        st.subheader("Личные данные")
        st.markdown(f"""
        **ФИО:** Фральцов Матвей Андреевич
        
        **Группа:** (ФИТ-231)
        
        **Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных
        """)
        
def show_dataset_page():
    st.title("Исследование набора данных (EDA)")

    with st.expander("Описание предметной области", expanded=True):
        st.write("""
        *Данный датасет содержит информацию о сыгранных раундах в компьютерной игре CS:GO*
        """)

    with st.expander("Таблица признаков"):
        features_data = {
            "time_left": ["Числовой", "время до конца раунда"],
            "ct_score": ["Числовой", "количество раундов, выигранных спецназом"],
            "t_score": ["Числовой", "количество раундов, выигранных террористами"],
            "map": ["Категориальный", "название карты"],
            "bomb_planted": ["Бинарный", "была ли установлена бомба"],
            "ct_health": ["Числовой", "сумма здоровья всех игроков спецназа"],
            "t_health": ["Числовой", "сумма здоровья всех игроков террористов"],
            "ct_armor": ["Числовой", "сумма показателей брони всех игроков спецназа"],
            "t_armor": ["Числовой", "сумма показателей брони всех игроков террористов"],
            "ct_money": ["Числовой", "сумма денег у команды спецназа"],
            "t_money": ["Числовой", "сумма денег у команды террористов"],
            "ct_helmets": ["Числовой", "у скольких игроков команды спецназа есть шлемы"],
            "t_helmets": ["Числовой", "у скольких игроков команды террористов есть шлемы"],
            "ct_defuse_kits": ["Числовой", "сколько у игроков спецназа есть наборов для обезвреживания бомбы"],
            "ct_players_alive": ["Числовой", "количество живых игроков спецназа"],
            "t_players_alive": ["Числовой", "количество живых игроков террористов"]
            }

        df_features = pd.DataFrame(features_data, index=["Тип данных", "Описание"]).T
        st.table(df_features)
            
    st.markdown("""
            **Особенности предобработки**   
            
            **Обработка пропусков:**
            - Было удалено незначительное количество строк с пропусками в столбцах `map` и `t_players_alive`.
            - Пропуски в столбце `t_health` были заполнены на основе столбца `t_players_alive`:
            - Если в строке **0 живых игроков**, то сумма здоровья равна **0**.
            - Если значение ненулевое, бралось **среднее** для того количества игроков, которое осталось в живых.
            - Пропуски в `ct_helmets`, `t_helmets`, `ct_defuse_kits` были заполнены **медианой**, основываясь на количестве брони.

            **Изменение типов данных:**
            - Столбец `bomb_planted` (тип `bool`) был заменен на `int`.
            - Большинство значений `float` были приведены к `int`, так как половинные значения отсутствуют.

            **Удаление аномалий:**
            - Были удалены аномалии из столбца `t_health`.
                """)

def show_visualize_page(df):
    st.title("Визуализация зависимостей в данных CS:GO")

    # time_left, ct_score, t_score, map, bomb_planted, ct_health, t_health, 
    # ct_armor, t_armor, ct_money, t_money, ct_helmets, t_helmets, 
    # ct_defuse_kits, ct_players_alive, t_players_alive

    tab1, tab2, tab3, tab4 = st.tabs([
        "Деньги и броня", 
        "Карты", 
        "Бомба и Время", 
        "Состояние команд"
    ])

    with tab1:
        st.subheader("Зависимость брони от наличия денег")
        fig1, ax1 = plt.subplots()
        sns.regplot(data=df, x='time_left', y='ct_money', ax=ax1, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        ax1.set_title("Спецназ (CT): Деньги vs Броня")
        st.pyplot(fig1)
        st.write("График показывает, при каком уровне бюджета команда начинает массово закупать броню.")

    with tab2:
        st.subheader("Распределение раундов по картам")
        fig2, ax2 = plt.subplots()
        df['map'].value_counts().plot(kind='bar', ax=ax2, color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Распределение времени в зависимости от установки бомбы")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='bomb_planted', y='time_left', ax=ax3, palette='Set3')
        st.pyplot(fig3)

    with tab4:
        st.subheader("Корреляция численного состава и здоровья")
        columns= ['time_left', 'bomb_planted',
                 'ct_players_alive', 't_players_alive']
        corr = df[columns].corr()
        
        fig4, ax4 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier


@st.cache_resource
def load_models():
    models = {}
    model_names = ["BaggingClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier", "StackingClassifier"]
    for name in model_names:
        try:
            with open(f"models/{name}.pkl", "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка загрузки {name}: {e}")
    
    try:
        models["CatBoost"] = CatBoostClassifier().load_model("models/CatBoostClassifier")
    except:
        st.warning("Модель CatBoost не найдена в папке models")
    return models

def show_prediction_page():
    st.title("Прогноз установки бомбы")

    models = load_models()
    if not models:
        st.error("Критическая ошибка: ни одна модель не загружена. Проверьте папку models.")
        return

    selected_model_name = st.selectbox("Выберите модель ML для прогноза", list(models.keys()))
    model = models[selected_model_name]

    tab_manual, tab_csv = st.tabs(["Ручной ввод параметров", "Массовый прогноз (CSV)"])

    with tab_manual:
        st.info("Введите текущие показатели раунда для мгновенного анализа.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Обстановка")
            map_name = st.selectbox("Карта (map)", ["de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train", "de_vertigo"])
            time_left = st.slider("Время до конца раунда (сек)", 0, 175, 60, help="Оставшееся время согласно таймеру игры")
            ct_score = st.number_input("Счет Спецназа (CT)", 0, 30, 0)
            t_score = st.number_input("Счет Террористов (T)", 0, 30, 0)
            ct_kits = st.number_input("Наборы сапера (defuse kits)", 0, 5, 2)

        with col2:
            st.subheader("Ресурсы команд")
            ct_health = st.slider("Суммарное HP Спецназа", 0, 500, 500)
            c_alive_max = min(5, max(0, ct_health // 1 + (1 if ct_health > 0 else 0)))
            ct_alive = st.number_input("Живых игроков CT", 0, 5, value=c_alive_max)
            ct_money = st.number_input("Бюджет CT ($)", 0, 50000, 16000)
            
            st.divider()
            
            t_health = st.slider("Суммарное HP Террористов", 0, 500, 500)
            t_alive_max = min(5, max(0, t_health // 1 + (1 if t_health > 0 else 0)))
            t_alive = st.number_input("Живых игроков T", 0, 5, value=t_alive_max)
            t_money = st.number_input("Бюджет T ($)", 0, 50000, 16000)

        error_flag = False
        if (t_alive > 0 and t_health == 0) or (ct_alive > 0 and ct_health == 0):
            st.error("Логическая ошибка: Кол-во живых игроков > 0, но суммарное здоровье равно 0.")
            error_flag = True
        if (t_alive == 0 and t_health > 0) or (ct_alive == 0 and ct_health > 0):
            st.warning("Внимание: Здоровье > 0, но живых игроков 0. Это может исказить прогноз.")

        if st.button("Рассчитать", disabled=error_flag):

            input_data = pd.DataFrame([{
                "time_left": time_left, "ct_score": ct_score, "t_score": t_score,
                "map": map_name, "ct_health": ct_health, "t_health": t_health,
                "ct_armor": 400 if ct_alive > 0 else 0, 
                "t_armor": 400 if t_alive > 0 else 0, 
                "ct_money": ct_money, "t_money": t_money,
                "ct_helmets": min(4, ct_alive), 
                "t_helmets": min(4, t_alive),
                "ct_defuse_kits": ct_kits, "ct_players_alive": ct_alive,
                "t_players_alive": t_alive
            }])


            map_mapping = {
                'de_inferno': 1, 'de_dust2': 2, 'de_nuke': 3, 'de_mirage': 4,
                'de_overpass': 5, 'de_train': 6, 'de_vertigo': 7, 'unknown': 8, 'de_cache': 9
            }
            input_data['map'] = input_data['map'].map(map_mapping)

            res = model.predict(input_data)[0]
            
            st.divider()
            
            if res == 1:
                st.error("### Прогноз: БОМБА БУДЕТ УСТАНОВЛЕНА")
            else:
                st.success("### Прогноз: Бомба НЕ будет установлена")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data)[0]
                prob_yes = probs[1]
                prob_no = probs[0]

                m1, m2 = st.columns(2)
                m1.metric("Шанс установки", f"{prob_yes*100:.1f}%")
                m2.metric("Шанс защиты", f"{prob_no*100:.1f}%")
                
                bar_color = "red" if prob_yes > 0.5 else "green"
                st.write(f"Уровень угрозы установки:")
                st.progress(prob_yes)
                
                if prob_yes > 0.75:
                    st.warning("**Высокая вероятность**")
                elif prob_yes < 0.25:
                    st.info("**Низкая вероятность** ")
                else:
                    st.write("**Ситуация 50/50**")

    with tab_csv:
        st.subheader("Массовая обработка данных")
        uploaded_file = st.file_uploader("Загрузите CSV с данными", type="csv")
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            X_test = data.drop(columns=['bomb_planted'], errors='ignore') 
            
            if X_test['map'].dtype == 'object':
                X_test['map'] = X_test['map'].map(map_mapping)

            preds = model.predict(X_test)
            data['bomb_planted_prediction'] = ["Да" if x == 1 else "Нет" for x in preds]
            
            st.write("### Результаты предсказаний:")
            st.dataframe(data, use_container_width=True)
            
            csv_result = data.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать CSV с прогнозами", csv_result, "predictions.csv", "text/csv")

page = st.sidebar.selectbox("Навигация", ["Об авторе",
                                          "Датасет и EDA",
                                          "Визуализация зависимостей",
                                          "Предсказание соответствующей  модели  ML"])

if page == "Информация о разработчике":
    show_about_page()
elif page == "Информация о датасете":
    show_dataset_page()
elif page == "Визуализация данных":
    show_visualize_page(df)
elif page == "Предсказания моделей":
    show_prediction_page()
