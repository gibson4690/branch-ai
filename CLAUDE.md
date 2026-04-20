# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install dependencies once:
```bash
python3 -m pip install -r requirements.txt --break-system-packages
```

## Running the App

```bash
# Development
bash entrypoint.sh

# Production
bash entrypoint.sh production
```

The Streamlit app listens on `0.0.0.0:8080`.

## Architecture

This is a Streamlit app:

- `app.py` — main application entry point
- `requirements.txt` — Python dependencies (`streamlit`)
- `entrypoint.sh` — runs `streamlit run app.py` with appropriate flags for dev/prod

## App Design

Name of app: Branch.ai

This app provides an intuitive user interface designed for Bank's Country Managers to gain insights from their business data without need to build and understand dashboards and databases. 

Theme: Light theme, professional and modern. Use emoticons sparingly.

In the main page, there must be a welcome banner (transparent, without border). Place the name of the app here, using a sleek and modern logo for it. Use a mix of colours and font to make it aesthetically pleasing and theme fitting. The banner should be flushed to the top and stay there even when the page scrolls.

### Highlights

Underneath the chat box is one container showing top positive highlights and another container showing top negative highlights of branch performance improvements. Both containers combined must be 60% page width.

Each container consists of list of insights cards. Each Insight card provides a short summary of the finding and a small graphical plot of the finding.

The insight cards should appear 3D with slight shadow. The entire card should be clickable. Do not use a standard button within each card.

### Chat Box

Underneath the highlights is a chat box (having 60% page width) for the user to ask questions about the data. Use professional icons that are not from the streamlit default. Ensure the UI are properly working, modern and sleek.

### Deep Dive Analysis

Upon user submission of the business question or if an insight card is clicked, scroll down to the Chat Box where a deep dive analysis into the submitted business question or the insight card will be provided in a streaming fashion. Use a LLM to analyse the question and data to answer the question. Use LLM to decide and dynamically produce plots that help visualise the findings. Use a plot catalogue and plot tool to support this feature.

The provided analysis should be concise and professional, driven purely by the data provided, and catered to senior management. 

Ensure the formatting is professional. Bullet points should have correct indentations and line spacing.

LLM to provide 3 suggested follow up questions, to be shown in the Suggestion Questions component.

### Suggested Questions
Suggest 3 short questions to ask. For each question, make it a clickable button that will populate the chatbox and send the question when clicked. The buttons should be highlighted when the mouse hovers above it.

If no questions have been asked yet, show 3 default short questions. Otherwise, use LLM's suggestion from the latest chat response. 

## Data

Create mock data used for the app. Assume the bank is UOSB, a local bank in Singapore. Data range from Jan 2024 to Mar 2026. Data include performance related metrics such as average wait time and handling times. Queue data such as queue token count and missed queue count. Demand side metrics such as transactions and customer demographics and types (e.g. customer aged 60, corporate clients). Supply side metrics such as counter utilisation and staff count by staff seniority (seedling, sapling, mature tree).


