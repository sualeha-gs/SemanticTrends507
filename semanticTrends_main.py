# to run: python -m streamlit run semanticTrends_main.py

import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
from itertools import islice

# Load graphs
@st.cache_data

# load graph files and keep them seperated by type
def load_graphs() -> dict:
    """
    Loads embedding and co-occurrence graphs for each year from 2015 to 2023.

    Reads GraphML files from the 'graphfiles' directory and parses them into NetworkX graphs.
    Adds a 'weight' attribute to each edge (defaulting to 'd0' or 1 if missing).
    Returns a dictionary of graphs keyed by their type and year (e.g., 'embedding_2018').

    Returns:
        dict[str, networkx.Graph]: A dictionary containing graphs for each year and type.
    """
    graphs = {}

    for year in range(2015, 2024):

        for kind, prefix in [("embedding", "graph"), ("cooccurrence", "cgraph")]:

            path = f"graphfiles/{prefix}_{year}.graphml"

            try:
                G = nx.read_graphml(path)
                for u, v, d in G.edges(data=True):
                    d['weight'] = float(d.get('weight', d.get('d0', 1)))
                
                graphs[f"{kind}_{year}"] = G

            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")

    return graphs

# plots to visualise word communities for 2 years side by side for easy comparison
def plot_side_by_side(G1: nx.Graph, G2: nx.Graph, title1: str, title2: str) -> None:
    """
    Plots two NetworkX graphs side by side for visual comparison.

    Args:
        G1 (networkx.Graph): The first graph to plot.
        G2 (networkx.Graph): The second graph to plot.
        title1 (str): Title for the first graph.
        title2 (str): Title for the second graph.

    Displays:
        A matplotlib figure with both graphs drawn side by side in the Streamlit app.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for ax, G, title in zip(axs, [G1, G2], [title1, title2]):
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, ax=ax, node_size=600, font_size=8)
        ax.set_title(title)
    
    # display fig in st
    st.pyplot(fig)

# ------------------------------------------------------------------------------------------------------------
# intro to app stuff - description and graph type selection

## heading
st.set_page_config(page_title="Semantic Trends", layout="centered")
st.title("Welcome to the Semantic Trend Analysis Tool!")

## explanation/intro stuff
st.markdown("""
This app lets you compare language trends and track semantic shifts based on Twitter data from **2015 to 2023**.

There are two types of graphs:
- **Embedding-based Graphs**: link semantically similar words to track semantic similarity
- **Co-occurrence Graphs**: link words that appear together in tweets to capture contextual association

How to use the app:
- For the type of analysis you are interested in, you can pick 2 years at a time to compare results. 
- For word associations or simlar word groups, enter the word you are interested in. 
- To shift to a different graph to change what you are analysing (i.e. you are looking at semantic realtionships (embeddings) but want to switch to associations (contextual)) simply use the toggle to change the graph type.           
""")

## load graphs 
graphs = load_graphs()
years = list(range(2015, 2024))

## have user pick analysis type
st.subheader("First pick what kind of analysis you are interested in")
graph_type_label = st.radio("Select graph type:", ["Embedding Graph - Semantic Similarity", "Co-occurrence Graph - Contextual Association"])
graph_type = "embedding" if "Embedding" in graph_type_label else "cooccurrence"

# ------------------------------------------------------------------------------------------------------------
# community clusters 
st.markdown("---")
st.subheader("Looking at the Top 3 Community Clusters")

## have user pick their years
st.markdown("Pick which two years you'd like to look at and compare:")
y1 = st.selectbox("Select Year 1", years, index=0, key = "y1_cluster")
y2 = st.selectbox("Select Year 2", years, index=1, key = "y2_cluster")

## get the graphs based on user's selection
G1 = graphs.get(f"{graph_type}_{y1}")
G2 = graphs.get(f"{graph_type}_{y2}")

if G1 is None or G2 is None:
    st.error("One of the selected graphs could not be loaded. Please check the graph files.")
    st.stop()

## show top 3 word community clusters based on greedy modularity
comms1 = list(islice(greedy_modularity_communities(G1, weight='weight'), 3))
comms2 = list(islice(greedy_modularity_communities(G2, weight='weight'), 3))

cols = st.columns(2)
for i in range(3):
    with cols[0]:
        st.write(f"**Year {y1} - Community {i+1}**: {sorted(list(comms1[i]))[:15]}")
    with cols[1]:
        st.write(f"**Year {y2} - Community {i+1}**: {sorted(list(comms2[i]))[:15]}")

for i in range(3):
    ### get only top 15 word in the plot so things stay readible
    top_nodes1 = sorted(comms1[i], key=lambda n: G1.degree(n), reverse=True)[:15]
    top_nodes2 = sorted(comms2[i], key=lambda n: G2.degree(n), reverse=True)[:15]
    
    ### create sub graohs containing only top nodes
    subG1 = G1.subgraph(top_nodes1)
    subG2 = G2.subgraph(top_nodes2)

    ### make plot
    plot_side_by_side(subG1, subG2, f"Year {y1} - Community {i+1}", f"Year {y2} - Community {i+1}")

# ------------------------------------------------------------------------------------------------------------
# interactive word similarity comparison
st.markdown("---")
st.subheader("Word Similarity Lookup")

## have user pick their years
st.markdown("Pick which two years you'd like to look at and compare:")
sy1 = st.selectbox("Select Year 1", years, index=0, key = "y1_sim")
sy2 = st.selectbox("Select Year 2", years, index=1, key = "y2_sim")

## get the graphs based on user's selection
G1 = graphs.get(f"{graph_type}_{sy1}")
G2 = graphs.get(f"{graph_type}_{sy2}")

query_word = st.text_input("Enter a word to compare its neighborhood across the two years")
if query_word:
    def get_neighbors(G, word):
        return sorted(G.neighbors(word)) if word in G else []

    neighbors1 = get_neighbors(G1, query_word)
    neighbors2 = get_neighbors(G2, query_word)

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{query_word} in {sy1}**: {neighbors1 if neighbors1 else 'Not found'}")
    with c2:
        st.write(f"**{query_word} in {sy2}**: {neighbors2 if neighbors2 else 'Not found'}")

# ------------------------------------------------------------------------------------------------------------
# centrality comparison plots for selected years
st.markdown("---")
st.subheader("Top 20 Central Words (Degree Centrality)")

## have user pick their years
st.markdown("Pick which two years you'd like to look at and compare:")
cy1 = st.selectbox("Select Year 1", years, index=0, key = "y1_cen")
cy2 = st.selectbox("Select Year 2", years, index=1, key = "y2_cen")

## get the graphs based on user's selection
G1 = graphs.get(f"{graph_type}_{cy1}")
G2 = graphs.get(f"{graph_type}_{cy2}")

## get top words based on how connected they are to other words
def get_top_degree_words(G, top_n=20):
    return sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:top_n]

top1 = get_top_degree_words(G1)
top2 = get_top_degree_words(G2)

## storage fro plotting
df1 = pd.DataFrame(top1, columns=["Word", "Degree"])
df2 = pd.DataFrame(top2, columns=["Word", "Degree"])

## display plots side by side for easy comparison
c1, c2 = st.columns(2)
with c1:
    st.write(f"**{cy1}**")
    st.bar_chart(df1.set_index("Word"))
with c2:
    st.write(f"**{cy2}**")
    st.bar_chart(df2.set_index("Word"))

# ------------------------------------------------------------------------------------------------------------
# emergent words
st.markdown("---")
st.subheader("Emergent Words")

## have user pick their years
st.markdown("Pick which two years you'd like to look at and compare:")
ey1 = st.selectbox("Select Year 1", years, index=0, key = "y1_em")
ey2 = st.selectbox("Select Year 2", years, index=1, key = "y2_em")

## get the graphs based on user's selection
G1 = graphs.get(f"{graph_type}_{ey1}")
G2 = graphs.get(f"{graph_type}_{ey2}")

## pick how many words to display
# default value to start to avoid an error showing up
num = st.text_input(f"To see the top x emergent words that started appearing in the {ey2} vocabulary since {ey1}, please pick a value for x: ", value="5")

st.markdown(f"Top {num} Emergent Words in Later Year:")

## set diff for words in y2 not in y1 - limited by the way the vocab is set up though
emergent_words = set(G2.nodes()) - set(G1.nodes())

## display results based on if words exist 
if emergent_words:
    st.write(f"Top {num} emergent words in {ey2} not seen in {ey1}:")
    st.write(sorted(emergent_words)[:int(num)])
else:
    st.write("No emergent words found.")
