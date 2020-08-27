mkdir -p ~/.streamlit/  

echo "\
[general]\n\
email = \"priyanhiman@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\                       
[server]\n\                       
port = 5000\n\                       
enableCORS = false\n\                       
headless = true\n\                       
\n\                       
" > ~/.streamlit/config.toml