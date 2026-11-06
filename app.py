import logging
import json
import streamlit as st
from chatbot import fresh_state, handle, refresh_policy
from engine import Engine

logging.basicConfig(
    level=logging.INFO, 
    format='%(name)s %(levelname)s %(message)s'
)
st.set_page_config(
    page_title='ReturnRisk', 
    page_icon='↩', 
    layout='centered'
)

st.title('↩ ReturnRisk')
st.caption('Understand return policies using traceable official evidence.')

if 'conversation' not in st.session_state:
    st.session_state.conversation = fresh_state()
    st.session_state.engine = Engine()

state = st.session_state.conversation
new, refresh = st.columns(2)

if new.button('New Conversation', use_container_width=True):
    st.session_state.conversation = fresh_state()
    st.session_state.engine = Engine()
    st.rerun()

# Refresh the policy evidence only when the retailer is known and the user clicks the button
if refresh.button('Refresh Policy', disabled=not bool(state[state['active']].retailer), use_container_width=True):
    with st.spinner('Refreshing official return-policy evidence…'):
        refresh_policy(state, st.session_state.engine)

#shows the conversation history + users latest message + assistant's latest response
def show(entry):
    st.markdown(entry['text'])
    if entry.get('comparison'):
        st.dataframe(entry['comparison'], hide_index=True, use_container_width=True)
        
    for source in entry.get('sources', []):
        with st.expander('Official policy quotation'):
            st.write(source['quote'])
            st.link_button('View official policy', source['url'])
    if entry.get('notes'):
        with st.expander('Conditions and refund details'):
            for note in entry['notes']:
                st.write(note)
    if entry.get('diagnostics'):
        with st.expander('Technical diagnostics'):
            st.code(json.dumps(entry['diagnostics'], indent=2), language='json')
            
for entry in state['history']:
    with st.chat_message(entry['role']):
        show(entry)
if (message := st.chat_input('Describe the product, retailer, purchase dates, and condition')):
    with st.chat_message('user'):
        st.write(message)
    with st.chat_message('assistant'):
        with st.spinner('Checking the request and relevant official evidence…'):
            reply = handle(message, state, st.session_state.engine)
        show(reply)
