let sidebarVisible = true;

const textarea = document.getElementById("userInput");
textarea.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
});

function handleKeyPress(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebarVisible = !sidebarVisible;

    if (sidebarVisible) {
        sidebar.classList.remove("hidden");
    } else {
        sidebar.classList.add("hidden");
    }
}

async function sendMessage() {
    const input = document.getElementById("userInput");
    const message = input.value.trim();

    if (!message) return;

    input.value = "";
    input.style.height = "auto";

    const welcomeMessage = document.querySelector(".welcome-message");
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    addMessage(message, "user");

    showLoading();

    scrollToBottom();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        hideLoading();

        if (data.status === "success") {
            addMessage(data.message, "bot", data.ml_insights);

            updateMLInsights(data.ml_insights);

            showSuggestions(data.ml_insights.suggestions);
        } else {
            addMessage("Sorry, may error: " + data.message, "bot");
        }

        scrollToBottom();
    } catch (error) {
        hideLoading();
        addMessage("Sorry, may problema sa connection. Please check your API key at internet connection.", "bot");
        scrollToBottom();
    }
}

function addMessage(text, sender, mlData = null) {
    const chatMessages = document.getElementById("chatMessages");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = sender === "user" ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;

    if (sender === "user" && mlData) {
        const mlBadge = document.createElement("div");
        mlBadge.className = "ml-badge";
        mlBadge.innerHTML = `
            <i class="fas fa-brain"></i> 
            ${mlData.sentiment.label} · ${mlData.intent}
        `;
        content.appendChild(mlBadge);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
}

function updateMLInsights(insights) {
    const sentiment = insights.sentiment;
    const sentimentEmoji = document.getElementById("sentimentEmoji");
    const sentimentLabel = document.getElementById("sentimentLabel");
    const sentimentScore = document.getElementById("sentimentScore");

    const emojiMap = {
        positive: "😊",
        negative: "😔",
        neutral: "😐"
    };

    sentimentEmoji.textContent = emojiMap[sentiment.label] || "😐";
    sentimentLabel.textContent = sentiment.label.charAt(0).toUpperCase() + sentiment.label.slice(1);
    sentimentScore.textContent = `Score: ${sentiment.score.toFixed(2)}`;
    sentimentScore.style.color = sentiment.label === "positive" ? "#28a745" : sentiment.label === "negative" ? "#dc3545" : "#6c757d";

    const intentBadge = document.getElementById("intentBadge");
    intentBadge.textContent = insights.intent;

    const topicsDisplay = document.getElementById("topicsDisplay");
    if (insights.topics && insights.topics.length > 0) {
        topicsDisplay.innerHTML = insights.topics.map(topic => `<span class="topic-tag">${topic}</span>`).join("");
    } else {
        topicsDisplay.innerHTML = '<span class="topic-tag">General</span>';
    }

    if (insights.user_insights) {
        const ui = insights.user_insights;
        document.getElementById("messageCount").textContent = ui.total_interactions;

        const moodTrend = document.getElementById("moodTrend");
        moodTrend.textContent = ui.sentiment_trend.charAt(0).toUpperCase() + ui.sentiment_trend.slice(1);
        moodTrend.style.color = ui.sentiment_trend === "positive" ? "#28a745" : ui.sentiment_trend === "negative" ? "#dc3545" : "#6c757d";

        const interestsList = document.getElementById("interestsList");
        if (ui.top_interests && ui.top_interests.length > 0) {
            interestsList.innerHTML = ui.top_interests
                .map(
                    (interest, index) =>
                        `<div class="interest-item">
                    <span>${interest}</span>
                    <span style="color: #667eea; font-weight: 600;">#${index + 1}</span>
                </div>`
                )
                .join("");
        }
    }
}

function showSuggestions(suggestions) {
    const suggestionsArea = document.getElementById("suggestionsArea");
    const suggestionsList = document.getElementById("suggestionsList");

    if (suggestions && suggestions.length > 0) {
        suggestionsList.innerHTML = suggestions.map(suggestion => `<div class="suggestion-chip" onclick="useSuggestion('${suggestion.replace(/'/g, "\\'")}')">${suggestion}</div>`).join("");
        suggestionsArea.style.display = "block";
    } else {
        suggestionsArea.style.display = "none";
    }
}

function useSuggestion(text) {
    document.getElementById("userInput").value = text;
    document.getElementById("userInput").focus();
}

function showLoading() {
    const chatMessages = document.getElementById("chatMessages");
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message bot-message";
    loadingDiv.id = "loadingMessage";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const loadingIndicator = document.createElement("div");
    loadingIndicator.className = "message-content";
    loadingIndicator.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    loadingDiv.appendChild(avatar);
    loadingDiv.appendChild(loadingIndicator);
    chatMessages.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingMessage = document.getElementById("loadingMessage");
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

function scrollToBottom() {
    const chatMessages = document.getElementById("chatMessages");
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function clearChat() {
    if (confirm("Are you sure you want to clear the chat and ML data?")) {
        try {
            await fetch("/clear", {
                method: "POST"
            });

            const chatMessages = document.getElementById("chatMessages");
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <i class="fas fa-brain"></i>
                    <h2>Welcome to AI ChatBot with ML!</h2>
                    <p>I'm powered by GPT and Machine Learning</p>
                    <div class="features">
                        <span><i class="fas fa-check"></i> Sentiment Analysis</span>
                        <span><i class="fas fa-check"></i> Intent Detection</span>
                        <span><i class="fas fa-check"></i> Smart Learning</span>
                    </div>
                </div>
            `;

            document.getElementById("sentimentEmoji").textContent = "😐";
            document.getElementById("sentimentLabel").textContent = "Neutral";
            document.getElementById("sentimentScore").textContent = "0.0";
            document.getElementById("intentBadge").textContent = "General";
            document.getElementById("topicsDisplay").innerHTML = '<span class="topic-tag">No topics yet</span>';
            document.getElementById("messageCount").textContent = "0";
            document.getElementById("moodTrend").textContent = "-";
            document.getElementById("interestsList").innerHTML = "<em>Start chatting to discover your interests!</em>";
            document.getElementById("suggestionsArea").style.display = "none";
        } catch (error) {
            alert("Error clearing chat");
        }
    }
}

async function showInsights() {
    try {
        const response = await fetch("/insights");
        const data = await response.json();

        if (data.status === "success" && data.insights) {
            const modal = document.getElementById("insightsModal");
            const modalBody = document.getElementById("modalBody");

            const insights = data.insights;

            modalBody.innerHTML = `
                <div style="text-align: center; margin-bottom: 30px;">
                    <h3 style="color: #667eea; margin-bottom: 20px;">
                        <i class="fas fa-chart-line"></i> Your Conversation Analytics
                    </h3>
                </div>
                
                <div class="insight-card">
                    <h3><i class="fas fa-comments"></i> Conversation Stats</h3>
                    <p><strong>Total Messages:</strong> ${insights.total_interactions}</p>
                    <p><strong>Sentiment Trend:</strong> <span style="color: ${insights.sentiment_trend === "positive" ? "#28a745" : insights.sentiment_trend === "negative" ? "#dc3545" : "#6c757d"}; font-weight: 600;">${insights.sentiment_trend}</span></p>
                    <p><strong>Average Sentiment Score:</strong> ${insights.average_sentiment.toFixed(2)}</p>
                </div>
                
                <div class="insight-card">
                    <h3><i class="fas fa-heart"></i> Your Top Interests</h3>
                    ${
                        insights.top_interests && insights.top_interests.length > 0
                            ? `<ul style="list-style: none; padding: 0;">
                            ${insights.top_interests
                                .map(
                                    (interest, i) =>
                                        `<li style="padding: 10px; background: ${i % 2 === 0 ? "#f8f9fa" : "white"}; margin: 5px 0; border-radius: 8px;">
                                    <i class="fas fa-star" style="color: #ffc107;"></i> ${interest}
                                </li>`
                                )
                                .join("")}
                        </ul>`
                            : "<p>No interests detected yet. Keep chatting!</p>"
                    }
                </div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <p style="font-size: 12px; color: #666;">
                        <i class="fas fa-info-circle"></i> These insights are generated using machine learning algorithms
                    </p>
                </div>
            `;

            modal.classList.add("active");
        } else {
            alert("No insights available yet. Start chatting to generate insights!");
        }
    } catch (error) {
        alert("Error loading insights");
    }
}

function closeModal() {
    document.getElementById("insightsModal").classList.remove("active");
}

async function exportData() {
    try {
        const response = await fetch("/export");
        const data = await response.json();

        if (data.status === "success") {
            const dataStr = JSON.stringify(data.data, null, 2);
            const dataBlob = new Blob([dataStr], { type: "application/json" });
            const url = URL.createObjectURL(dataBlob);

            const link = document.createElement("a");
            link.href = url;
            link.download = `chatbot_data_${new Date().getTime()}.json`;
            link.click();

            URL.revokeObjectURL(url);

            alert("Data exported successfully!");
        }
    } catch (error) {
        alert("Error exporting data");
    }
}

document.getElementById("insightsModal").addEventListener("click", function (e) {
    if (e.target === this) {
        closeModal();
    }
});

window.addEventListener("load", function () {
    document.getElementById("userInput").focus();

    if (window.innerWidth <= 1024) {
        toggleSidebar();
    }
});

window.addEventListener("resize", function () {
    if (window.innerWidth <= 1024 && sidebarVisible) {
        toggleSidebar();
    }
});
