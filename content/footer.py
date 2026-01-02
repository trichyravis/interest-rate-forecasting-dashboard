import streamlit as st

def display_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="color: #002147; margin-bottom: 10px;">🏔️ THE MOUNTAIN PATH - WORLD OF FINANCE</h4>
                <p style="font-size: 0.9rem; color: gray;">Bridging Academic Theory with Institutional Practice</p>
                <div style="display: flex; justify-content: center; gap: 15px; margin-top: 15px;">
                    <a href="https://www.linkedin.com/in/trichyravis" target="_blank">
                        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
                    </a>
                    <a href="https://github.com/trichyravis" target="_blank">
                        <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
                    </a>
                </div>
                <p style="margin-top: 20px; font-size: 0.8rem; color: #888;">
                    © 2026 The Mountain Path | All Rights Reserved | Educational Purpose Only
                </p>
            </div>
        """, unsafe_allow_html=True)
