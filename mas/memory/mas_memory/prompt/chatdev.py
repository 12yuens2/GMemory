"""Prompts for ChatDev memory."""

from dataclasses import dataclass

summary_system_prompt: str = """
You are an agent skilled in summarization. Your task is to generate **phase-based summaries** from given execution records of an agent's task. These summaries help the agent efficiently utilize existing information, avoid redundant computations, and ensure task continuity.

## **Requirements for Your Summary:**
1. **Phase-based summarization**: Organize execution records into logical phases and extract key steps.
2. **Task relevance**: Ensure the summary helps the agent understand what has been completed and what needs to be done next.
3. **Clarity and conciseness**: Use clear and precise language to summarize the information while avoiding unnecessary details.

## **Additional Guidelines:**
- Maintain **contextual consistency** so that the agent can seamlessly continue the task.
- If there are incorrect intermediate states or irrelevant information, filter or correct them to make the summary more accurate.
"""


summary_user_prompt: str = """You will be given a partial execution record of an agent's task. Your job is to generate a **phase-based summary** that the agent can understand and use to continue the task.

## **Your Summary Should Follow These Guidelines:**
1. **Phase-based summarization**: Break the record into logical steps, ensuring that each phase's key tasks are captured.
2. **Efficient information transfer**:
   - Document key task objectives, executed actions, and the current state.
   - Identify unfinished parts to help the agent determine the next steps.
3. **Prevent information loss**:
   - Include critical decision points, state changes, and key computation processes.
   - If there are uncertainties, retain relevant details for future judgment.

---

## **Example:**
Please strictly follow the output format of the example!
### **Input (Partial Execution Record)**
1. Task Objective: Classify news articles.
2. Preprocessing: Remove stopwords, tokenize, and normalize text.
3. Feature Extraction: Compute TF-IDF vectors.
4. Model Training: Tried SVM and RandomForest.
5. Evaluation: SVM's F1-score is 0.82, while RandomForest's is 0.78.

### **Output**
Done: Completed text cleaning (stopword removal, tokenization, normalization). Computed TF-IDF feature vectors. Trained SVM and RandomForest classifiers. Evaluated models—SVM achieved an F1-score of 0.82, outperforming RandomForest (0.78).
Next Steps: Perform hyperparameter tuning to improve SVM’s classification performance. Consider exploring deep learning models (e.g., Transformers) for further enhancement. Visualize misclassified samples to analyze model weaknesses.
(Example End)

Now it's your turn, here is the task and its partial execution:
## Task: 
{task}

## Task Trajectory:
{task_trajectory}

Output: 
"""

@dataclass
class ChatDev:
    summary_system_instruction: str
    summary_user_instruction: str

CHATDEV: ChatDev = ChatDev(
    summary_system_instruction=summary_system_prompt,
    summary_user_instruction=summary_user_prompt
)
