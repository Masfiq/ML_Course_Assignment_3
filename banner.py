import matplotlib.pyplot as plt

topic_number = 0

def reset(topic_list):
    global topic_number, topics
    
    topic_number = 0
    topics = topic_list

    print('\n\nTopics in this Notebook\n')
    for i, top in enumerate(topics):
        print(f'{i+1}. {top}')

    
def next_topic():
    global topic_number
    
    font = {'family': 'serif',
        'color':  'darkblue',
        'weight': 'bold',
        'size': 30,
        }

    plt.subplots_adjust(top = .8, bottom = 0.75, right = 1, left = 0,   
                        hspace = 0, wspace = 0)
    plt.tight_layout()
    plt.axis('off')

    topic_number += 1
    
    # n_txt = f'\n {topic_number}. {topics[topic_number - 1]} \n'
    n_txt = f'\n {topics[topic_number - 1]} \n'
    plt.text(0.5, 0.5, n_txt, ha='center',  wrap=True,
             backgroundcolor='lightyellow', 
             fontdict=font,
             bbox=dict(facecolor='yellow',
                       edgecolor='blue',
                       linewidth=5))