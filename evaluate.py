#!/usr/bin/env python
import sys
import os
import os.path
import io
import csv
from bleuscore import calculate_bleu_score
from rougescore import calculate_rouge_scores
from chrfscore import calculate_chrf_score


#subprocess.run(["pip", "install", "rouge"])

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    truth_file = os.path.join(truth_dir, "truth.csv")
    submission_answer_file = os.path.join(submit_dir, "answer.csv")

    Rscore = calculate_rouge_scores(truth_file, submission_answer_file)
    Cscore = calculate_chrf_score(truth_file, submission_answer_file)
    Bscore = calculate_bleu_score(truth_file, submission_answer_file)

    rounded_Bscore = round(Bscore, 3)
    rounded_Rscore = round(Rscore, 3)
    rounded_Cscore = round(Cscore, 3) 

    total_score = rounded_Bscore + rounded_Rscore + rounded_Cscore
    rounded_total_score = round(total_score,3)

    output_filename = os.path.join(output_dir, 'scores.txt')
    with io.open(output_filename, "w", encoding='utf-8') as f:
        f.write("bleu_score: %f\n" % rounded_Bscore)
        f.write("rouge_score: %f\n" % rounded_Rscore)
        f.write("chrf_score: %f\n" % rounded_Cscore)
        f.write("total_score: %f\n" % rounded_total_score)

