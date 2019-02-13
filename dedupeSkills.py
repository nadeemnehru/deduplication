import csv
import dedupe

skills1_100 = []
count = 0

#Extract first 100 skills from the skills file
with open('./Downloads/Data/Datasets/NonMedical/skillsNew.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        skills1_100.append(row)
        count += 1
        if count == 101:
            break

# Write the extracted skills to a new file
with open('./Downloads/Data/Datasets/NonMedical/skills1_100.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(skills1_100)

# Relevant files for deduplication
input_file = './Downloads/Data/Datasets/NonMedical/skills1_100.csv'
output_file = './Downloads/Data/Datasets/NonMedical/skills1_100Output'
settings_file = './Downloads/Data/Datasets/NonMedical/skills1_100_learned_settings'
training_file = './Downloads/Data/Datasets/NonMedical/skills1_100_training.json'

#Create data dictionary for deduplication
i = 0
data_d = {}
with open(input_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        r = [(k, v) for (k, v) in row.items()]
        data_d[i] = dict(r)
        i += 1

#Specify fields to use for deduplication
fields = [
    {'field': 'Skill', 'type': 'String'}
]

#Create deduper object
deduper = dedupe.Dedupe(fields)

#Sample data for active learning
deduper.sample(data_d) 

#Label data for active learning
dedupe.consoleLabel(deduper)

#Learn the model
deduper.train()

#Write training and settings files
with open(training_file, 'w') as tf:
    deduper.writeTraining(tf)

with open(settings_file, 'wb') as sf:
    deduper.writeSettings(sf)


# Deduplicate using different recall weights
for weight in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
    threshold = deduper.threshold(data_d, recall_weight=weight)
    clustered_dupes = deduper.match(data_d, threshold)
    print('# duplicate sets', len(clustered_dupes))
    cluster_membership = {}
    cluster_id = 0
    for (cluster_id, cluster) in enumerate(clustered_dupes):
        id_set, scores = cluster
        cluster_d = [data_d[c] for c in id_set]
        canonical_rep = dedupe.canonicalize(cluster_d)
        for record_id, score in zip(id_set, scores):
            cluster_membership[record_id] = {
                "cluster id" : cluster_id,
                "canonical representation" : canonical_rep,
                "confidence": score
            }
    singleton_id = cluster_id + 1
    
    with open(output_file + 'recWt' + str(weight) + 'thr' + str(threshold) + '.csv', 'w') as f_output, open(input_file) as f_input:
        writer = csv.writer(f_output)
        reader = csv.reader(f_input)

        heading_row = next(reader)
        heading_row.insert(0, 'confidence_score')
        heading_row.insert(0, 'Cluster ID')
        canonical_keys = canonical_rep.keys()
        for key in canonical_keys:
            heading_row.append('canonical_' + key)

        writer.writerow(heading_row)

        row_id = 0
        for row in reader:
            if row_id in cluster_membership:
                cluster_id = cluster_membership[row_id]["cluster id"]
                canonical_rep = cluster_membership[row_id]["canonical representation"]
                row.insert(0, cluster_membership[row_id]['confidence'])
                row.insert(0, cluster_id)
                for key in canonical_keys:
                    row.append(canonical_rep[key].encode('utf8'))
            else:
                row.insert(0, None)
                row.insert(0, singleton_id)
                singleton_id += 1
                for key in canonical_keys:
                    row.append(None)
            row_id +=1
            writer.writerow(row)
