import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_graph(xlabel,ylabel,title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("Graphs\\NASA\\"+title+".png")
    #plt.show()


# Read CSV file
df = pd.read_csv('Csv_data\\mei_nasa_1993_cleaned.csv', delimiter=';')


"""
#######Jobs_per_month########

# Group data by year and month, count number of jobs
jobs_per_month = df.groupby(['year', 'month'])['job_number'].count().reset_index()

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(jobs_per_month)), jobs_per_month['job_number'], tick_label=jobs_per_month['month'])
plt.xticks(rotation=45)  # Rotate month labels for better visibility

show_graph("Month", "Number of Jobs", "Number of Jobs Per Month")


#######Jobs_per_day_of_week########

jobs_per_day_of_week = df.groupby('day_of_week')['job_number'].count().reset_index()

# Create a bar chart for jobs per day of the week
plt.figure(figsize=(8, 6))
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.bar(days, jobs_per_day_of_week['job_number'])

show_graph("Day of the Week", "Number of Jobs", "Number of Jobs Per Day of the Week")


#######Jobs_per_hour########

# Group data by hour, count number of jobs
jobs_per_hour = df.groupby('hour')['job_number'].count().reset_index()
hours = range(24)  # 24 hours in a day
plt.bar(hours, jobs_per_hour['job_number'])

show_graph("Hour", "Number of Jobs", "Number of Jobs Per Hour")


#######Jobs_per_day########

# Group data by day, count number of jobs
jobs_per_day = df.groupby('day')['job_number'].count().reset_index()
plt.bar(jobs_per_day['day'], jobs_per_day['job_number'])

show_graph("Day", "Number of Jobs", "Number of Jobs Per Day")



#######Jobs_per_user########

# Group data by user, calculate total runtime for each user
runtime_per_user = df.groupby('user')['runtime'].mean().reset_index()
# Create a bar chart for runtime per user
plt.figure(figsize=(10, 6))
plt.bar(runtime_per_user['user'], runtime_per_user['runtime'])

show_graph("User", "Average Runtime", "Average Runtime Per User")


#######Jobs_per_user_group########
# Group data by user group, calculate total runtime for each user group
runtime_per_user_group = df.groupby('user_group')['runtime'].mean().reset_index()

count = df.groupby('user_group')['job_number'].count().reset_index()
print(count)
# Create a bar chart for runtime per user group
plt.figure(figsize=(10, 6))

plt.bar(runtime_per_user_group['user_group'], runtime_per_user_group['runtime'])
show_graph("User Group", "Average Runtime", "Average Runtime Per User Group")



#######Jobs_per_user########
# Group data by user, count number of jobs for each user
jobs_per_user = df.groupby('user')['job_number'].count().reset_index()
# Create a bar chart for number of jobs per user
plt.figure(figsize=(10, 6))
plt.bar(jobs_per_user['user'], jobs_per_user['job_number'])
show_graph("User", "Number of Jobs", "Number of Jobs Per User")


#######Jobs_running_at_same_time########

import pandas as pd
import matplotlib.pyplot as plt


# Converter submit_time para objetos de datetime
df['submit_time'] = pd.to_datetime(df['submit_time'], unit='s')

# Extrair a hora do dia e o dia da semana (1 a 7, onde 1 é segunda-feira e 7 é domingo)
df['hour'] = df['submit_time'].dt.hour
df['weekday'] = df['submit_time'].dt.dayofweek + 1

# Classificar os dados por submit_time
df = df.sort_values(by='submit_time')

# Contar o número de submissões por hora durante os dias de semana e o fim de semana
submissions_weekday = df[df['weekday'] <= 5].groupby('hour')['job_number'].count()
submissions_weekend = df[df['weekday'] > 5].groupby('hour')['job_number'].count()

# Preparar rótulos para o eixo x (horas)
hour_labels = list(range(24))

# Plotar
plt.figure(figsize=(10, 6))
plt.plot(hour_labels, submissions_weekday, label='Dias de Semana', marker='o')
plt.plot(hour_labels, submissions_weekend, label='Fim de Semana', marker='o')

plt.xlabel('Hora do Dia')
plt.ylabel('Número de Submissões')
plt.title('Número de Submissões por hora (weekend vs weekday))')
plt.xticks(hour_labels)
plt.legend()
plt.tight_layout()

# Salvar o gráfico como arquivo PNG
#plt.savefig('submissoes_por_hora.png')
plt.show()







# Encontrar o índice onde o executable_num é máximo

# Contar a frequência de cada executable_num
executable_num_counts = df['executable_num'].value_counts()

# Selecionar os top 15 executable_nums
top_15_executable_num_counts = executable_num_counts.head(15)

# Gráfico de pizza (pie chart) com os top 15 executable_nums
plt.figure(figsize=(8, 8))
plt.pie(top_15_executable_num_counts, labels=top_15_executable_num_counts.index, autopct='%1.2f%%', startangle=140)
plt.title('Top 15 Executable_nums')
plt.tight_layout()
plt.savefig('top_15_executable_nums_pie_chart.png')
plt.show()

"""
"""

# Determine the total duration of your dataset
total_duration = df['submit_time'].max() + df['runtime'].max()

# Initialize an array to store the number of jobs running at each time unit
time_units = np.arange(0, total_duration)
job_counts = np.zeros(total_duration)

# Update the array to reflect the number of jobs running at each time unit
for _, row in df.iterrows():
    submit_time = int(row['submit_time'])
    runtime = int(row['runtime'])
    job_counts[submit_time:(submit_time + runtime)] += 1

# Calculate the mean of job counts
mean_job_count = np.mean(job_counts)

#compute the job_counts equal to 1
job_counts_equal_to_1 = np.count_nonzero(job_counts == 1)
#compute the job_counts bigger than 1
job_counts_bigger_than_1 = np.count_nonzero(job_counts > 1)
#compute the job_counts equal to 0
job_counts_equal_to_0 = np.count_nonzero(job_counts == 0)


#pie chart for the job_counts_equal_to_0 and job_counts_equal_to_1 + job_counts_bigger_than_1
labels = 'Downtime', 'Uptime'
sizes = [job_counts_equal_to_0, job_counts_equal_to_1 + job_counts_bigger_than_1]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, startangle=140)

plt.show()

sequencial_jobs = job_counts_equal_to_0 + job_counts_equal_to_1
paralel_jobs = job_counts_bigger_than_1

print("sequencial_jobs: ", sequencial_jobs/len(job_counts))
print("paralel_jobs: ", paralel_jobs/len(job_counts))

print(job_counts_equal_to_1/len(job_counts)), print(job_counts_bigger_than_1/len(job_counts)), print(job_counts_equal_to_0/len(job_counts))



# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(time_units, job_counts)
plt.xlabel('Time')
plt.ylabel('Number of Jobs Running')
plt.title('Number of Jobs Running at the Same Time')
plt.axhline(y=mean_job_count, color='r', linestyle='--', label=f'Mean: {mean_job_count:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('number_of_jobs_running_at_the_same_time.png')
plt.show()
"""

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the average runtime for each month
average_runtime_per_month = df.groupby('month')['runtime'].mean()

# Create a violin plot using seaborn
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.violinplot(x='month', y='runtime', data=df, palette='muted')

# Annotate the violin plot with average runtime for each month
for i, month in enumerate(average_runtime_per_month.index):
    avg_runtime = average_runtime_per_month.values[i]
    plt.text(i, avg_runtime, f'{avg_runtime:.2f}', ha='center', va='bottom', color='black')

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Runtime')
plt.title('Runtime per Month')

# Show the plot
plt.show()
data = df
# Group the data by 'user_group'
grouped_data = data.groupby('user_group')['nodes_alloc'].value_counts(normalize=True).unstack().fillna(0)

# Calculate the percentage distribution of nodes allocated per user group
percentage_distribution = grouped_data.apply(lambda x: x / x.sum(), axis=1) * 100

# Plotting
plt.figure(figsize=(10, 6))
percentage_distribution.plot(kind='bar', stacked=True)
plt.xlabel('User Group')
plt.ylabel('Percentage of Nodes Allocated')
plt.title('Percentage Distribution of Nodes Allocated per User Group')
plt.legend(title='Nodes Allocated', bbox_to_anchor=(1, 1))
plt.xticks(rotation=0)
plt.show()
data = df
# Group the data by 'hour' and sum the 'nodes_alloc' values
nodes_alloc_per_hour = data.groupby('hour')['nodes_alloc'].sum()

# Plotting
plt.figure(figsize=(10, 6))
nodes_alloc_per_hour.plot(kind='bar', color='skyblue')
plt.xlabel('Hour of Day')
plt.ylabel('Total Nodes Allocated')
plt.title('Nodes Allocated per Hour of the Day')
plt.xticks(rotation=0)
plt.show()

dados = df
# Calcular a média dos runtimes
media_runtime = dados['runtime'].mean()

# Imprimir a média dos runtimes
print('Média dos Runtimes:', media_runtime)
"""

"""
# Group the data by 'user' and sum the 'nodes_alloc' values
nodes_alloc_per_user = df.groupby('user')['nodes_alloc'].sum()

# Plotting
plt.figure(figsize=(10, 6))
nodes_alloc_per_user.plot(kind='bar', color='skyblue')
plt.xlabel('User')
plt.ylabel('Total Nodes Allocated')
plt.title('Total Nodes Allocated per User')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""

data = df

# Group data by user, count number of jobs and sum nodes allocated for each user
user_stats = data.groupby('user').agg({'job_number': 'count', 'nodes_alloc': 'sum'}).reset_index()

# Plotting
plt.figure(figsize=(12, 6))

# Plot number of jobs per user
plt.bar(user_stats['user'], user_stats['job_number'], label='Number of Jobs', color='skyblue', width=0.4)

# Plot total nodes allocated per user as bars next to job submissions
bar_width = 0.4
user_positions = [x + bar_width for x in range(len(user_stats['user']))]
plt.bar(user_positions, user_stats['nodes_alloc'], label='Total Nodes Allocated', color='orange', width=0.4)

plt.xlabel('User')
plt.ylabel('Count')
plt.title('Number of Jobs and Total Nodes Allocated Per User')
plt.xticks([r + bar_width/2 for r in range(len(user_stats['user']))], user_stats['user'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()