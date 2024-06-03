import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from util import get_random_topic, transform_json
import json
import os

st.set_page_config(page_title="상담봇", page_icon=":tada:", layout="wide")
st.title("PHQ-9 기반 우울증 상담-진단 BETA")

# langchain config
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
memory = StreamlitChatMessageHistory(key="chat_messages")

gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """ 
당신은 정신건강 질환 탐지를 위한 챗봇 '진단봇'입니다. 당신의 말투는 공손하고 공감할 수 있어야 합니다. 이전 질문과 답변을 참고하여 대화를 자연스럽게 이어가세요. 당신은 이전 대화 기록을 참조하여 current_question에 대해서 자연스러운 질문을 생성해야 합니다.
current_question은 user의 답변이 아닌, 당신이 생성해야 하는 질문입니다. 당신이 생성하는 질문은 대화를 가능한 한 오래 유지할 수 있도록 해야 합니다. 모든 답변은 한국어로 이루어져야 합니다.
         
당신이 이전 대화를 참조해서 생성해야 하는 질문의 예시들은 다음과 같습니다.
=======================================================
<이전 대화>
책은 뭐 읽진 않았고, 여가 활동 같은것도 잘 안 하고 그냥 걷기만 해.

<이전 대화를 참조한 생성해야 하는 자연스러운 질문>
아, 그렇군요. 요즘 걷는 것 외에는 다른 활동을 하기가 힘드신가 봐요. 혹시 걷는 것 외에 다른 활동을 하기 힘든 이유가 있으신가요? 예를 들어, 힘이 없거나, 집중이 잘 안 되거나, 아니면 다른 이유가 있으신가요?
=======================================================
<이전 대화>
안녕하세요! 저는 진단봇이라고 합니다. 😊 오늘 저와 함께 편안하게 이야기 나눠보시겠어요?
혹시 요즘, 정신 건강에 대해 생각해 보신 적 있으신가요? 🤔 저는 정신 건강이 얼마나 중요한지 항상 강조하고 싶어요. 어떤 점이 가장 중요하다고 생각하시나요?

정신 건강은 다른게 아니라 내 상태를 정확히 아는게 중요하다고 생각해.
         
<이전 대화를 참조한 생성해야 하는 자연스러운 질문>
맞아요. 정신 건강은 자신의 상태를 정확히 아는 것에서부터 시작된다고 생각해요. 😊 그렇다면, 요즘 본인의 상태를 어떻게 판단하고 계신가요? 혹시, 일이나 여가 활동을 할 때 흥미나 즐거움을 느끼지 못하시나요?
=======================================================
이 밑에는 대화 기록과 당신이 참조해서 생성해야 하는 설문지 질문이 존재합니다. User's input에 공감하면서 current survey question에 대해서 물어보는 식으로 예시를 보면서 생성하세요.

"""),
        MessagesPlaceholder(variable_name="history"),
         ("human", "User's input: {question} Current survey question: {current_question}. "),
    ]
)

chain = gen_prompt | model

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="history",
    other_inputs={"current_question": "current_question"},
)


# 설문지 질문 리스트
survey_questions = [
    "일 또는 여가 활동을 하는데 흥미나 즐거움을 느끼지 못함",
    "기분이 가라앉거나.우울하거나.희망이 없음",
    "잠이 들거나 계속 잠을 자는 것이 어려움. 또는 잠을 너무 많이 잠",
    "피곤하다고 느끼거나 기운이 거의 없음",
    "입맛이 없거나 과식을 함",
    "자신을 부정적으로 봄, 혹은 자신이 실패자라고 느끼거나 자신 또는 가족을 실망시킴",
    "신문을 읽거나 텔레비전 보는 것과 같은 일에 집중하는 것이 어려움",
    "다른 사람들이 주목할 정도로 너무 느리게 움직이거나 말을 함 또는 반대로 평상시보다 많이 움직여서, 너무 안절부절 못하거나 들떠 있음",
    "자신이 죽는 것이 더 낫다고 생각하거나 어떤 식으로든 자신을 해칠것이라고 생각함"
]

# markdown setup
st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "survey_index" not in st.session_state:
    st.session_state.survey_index = 0
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []
if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

# runner 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# session state main page
if not st.session_state.messages:
    memory.clear()
    persona_message = """당신은 정신건강 질환 탐지를 위한 챗봇 '진단봇'입니다. 모든 대화는 한국어로 이루어질 것입니다. 그리고 당신은 언제나 예의바르게 대화해야 합니다. 상담 절차는 다음과 같습니다. 
    처음에는 간단한 인사와 함께 자기 소개를 하면 됩니다.
    그리고, 심리 상담을 할 것을 알려주면 됩니다. 심리 상담은 총 9개의 설문으로 이루어져 있으며, 솔직하게 대답해 줄 것을 부탁합니다.
    다음으로, 심리 상담 이전에 간단한 대화를 해 볼 것을 권합니다. 이 때, 무작위로 선택된 환기용 대화 주제가 주어집니다. 이 질문으로 시작해 피상담인과의 대화가 이루어지고, 앞으로의 대화에 연결된 만큼, 친절하고 자연스럽게 시작하면 됩니다. 대화가 자연스럽게 이어질 수 있도록 하세요."""
    initial_human_message = f"Start the conversation with this topic: {get_random_topic()}"

    with st.chat_message("assistant"):
        response = st.write_stream(model.stream([persona_message, initial_human_message]))
    st.session_state.messages.append({"role": "assistant", "content": response})
    memory.add_message(response)

# Function to handle the conversation
def handle_conversation():
    if st.session_state.survey_index < len(survey_questions):
        current_question = survey_questions[st.session_state.survey_index]
    else:
        prompt = st.chat_input("Enter a prompt here", key="end_prompt")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = {
            "question": prompt,
            "current_question": "",
        }
        
        response = chain_with_history.invoke(context, config={"configurable": {"session_id": "any"}})
        
        # Append context and response to conversation log
        st.session_state.conversation_log.append({
            "context": context,
            "response": response.content
        })
        
        st.markdown("**모든 설문이 완료되었습니다. 감사합니다.**")

        save_conversation()
        eval_chain_action()
        return

    if prompt := st.chat_input("Enter a prompt here", key=f"prompt_{st.session_state.survey_index}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare the context for the current survey question
        context = {
            "question": prompt,
            "current_question": current_question,
        }
        
        response = chain_with_history.invoke(context, config={"configurable": {"session_id": "any"}})
        
        # Append context and response to conversation log
        st.session_state.conversation_log.append({
            "context": context,
            "response": response.content
        })
        
        with st.chat_message("assistant"):
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        memory.add_message(prompt)
        memory.add_message(response.content)

        if st.session_state.survey_index < len(survey_questions):
            st.session_state.survey_index += 1

def save_conversation():
    if st.session_state.conversation_log:
        with open("conversation_log.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversation_log, f, ensure_ascii=False, indent=4)

def eval_chain_action():
    if st.session_state.conversation_log and not st.session_state.evaluation_done:
        if os.path.exists("conversation_log.json"):
            with open("conversation_log.json", "r", encoding="utf-8") as file:
                data = json.load(file)

            evaluation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 정신건강 평가를 돕는 AI입니다. 사용자의 응답을 0점부터 4점까지의 스케일로 평가해주세요. 여기서 0점은 매우 긍정적인 점수, 4점은 매우 부정적인 점수입니다. 또한, 각 항목에 대한 평가는 질문지에 대해서 '우울증'척도로 계산 했을 때 기준입니다."),
                ("human", "AI의 질문: {context}\n 사용자의 응답: {response}\n")
            ])

            eval_chain = evaluation_prompt_template | model
            eval_response_list = []
            for log in transform_json(data):
                eval_context = {
                    "context": f"Survey_question: {log['context']['current_question']}, Generated_question: {log['context']['question']}",
                    "response": log["response"]
                }
                eval_response = st.write_stream(eval_chain.stream(eval_context, config={"configurable": {"session_id": "any"}}))
                eval_response_list.append(eval_response)
            
            score_chain = model.invoke(f"""이것은 우울증 검진에 따른 평가입니다.
                                        이 평가 자료의 점수를 보고, 총점이 몇점인지 계산하시오. 그리고 기준에 따라서 답변을 생성하시오. 
                                        <평가 자료>
                                        {eval_response_list}
                                        </평가 자료>
                                        <기준>
                                        0~4점 / 심각도 None / 정상 범위
                                        5~9점 / 심각도 mild / 경과 관찰
                                        10~14점 / 심각도 Moderate / 치료 고려, 경과 관찰
                                        15~19점 / 심각도 Moderately Severe / 치료 요함(약물, 상담)
                                        20~27점 / 심각도 Severe / 적극적인 치료, 정신과 진료 필요.
                                        </기준>
                                    """)

            st.markdown(score_chain.content)
            # status flag
            st.session_state.evaluation_done = True

# main
handle_conversation()

# sidebar setup
def reset_conversation():
    st.session_state.messages = []
    st.session_state.survey_index = 0
    st.session_state.conversation_log = []
    st.session_state.evaluation_done = False
    st.session_state.clear()

with st.sidebar:
    st.sidebar.title("Options")
    st.button('Reset Chat', on_click=reset_conversation, use_container_width=True)
