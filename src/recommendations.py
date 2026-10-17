"""
Academic Support Recommendations Module for EduPredict AI
Generates rule-based, transparent, and actionable educational suggestions
based on key academic, lifestyle, and demographic indicators.

Note:
These recommendations are heuristic suggestions designed to guide educational
counselors and students. They do NOT represent automated causal claims of the ML model.
"""

from typing import Any, Dict, List


def generate_academic_recommendations(
    student_data: Dict[str, Any],
    predicted_category: str = ""
) -> Dict[str, Any]:
    """
    Generate prioritized academic support suggestions based on observable indicators.
    
    Args:
        student_data: Dict of student attributes (studytime, failures, G1, G2, absences, etc.)
        predicted_category: Model's predicted performance tier.
        
    Returns:
        Dict containing categorized recommendations (urgent, general, positive)
        along with explanatory disclaimer.
    """
    urgent_actions: List[Dict[str, str]] = []
    recommended_actions: List[Dict[str, str]] = []
    positive_notes: List[Dict[str, str]] = []

    # 1. Previous Academic Grades (G1, G2 / Mid-Terms / Sessional Tests)
    g1 = student_data.get("G1")
    g2 = student_data.get("G2")

    if g1 is not None and g2 is not None:
        try:
            g1_val = float(g1)
            g2_val = float(g2)

            if g1_val < 10 or g2_val < 10:
                urgent_actions.append({
                    "indicator": "Low Mid-Term Assessment Scores (Below Passing Threshold of 10/20)",
                    "title": "Immediate Targeted Remediation for End Semester Exam",
                    "suggestion": (
                        f"Prior assessment scores (Mid-Term 1={g1_val}, Mid-Term 2={g2_val}) fall below the standard passing score. "
                        "Schedule weekly review sessions focusing on fundamental prerequisite concepts and conduct "
                        "practice problem sets before the End Semester Exam."
                    ),
                    "action_tag": "Academic Intervention"
                })
            elif g2_val < g1_val - 2:
                recommended_actions.append({
                    "indicator": "Declining Performance Trajectory (Mid-Term 2 < Mid-Term 1)",
                    "title": "Identify Recent Concept Roadblocks",
                    "suggestion": (
                        f"Performance dropped from Mid-Term 1 ({g1_val}) to Mid-Term 2 ({g2_val}). Meet with the course instructor "
                        "to pinpoint difficult intermediate topics before they affect the End Semester Exam."
                    ),
                    "action_tag": "Grade Trend Alert"
                })
            elif g2_val >= 14 and g1_val >= 14:
                positive_notes.append({
                    "indicator": "Strong Academic Consistency",
                    "title": "Honors & Advanced Topic Exploration",
                    "suggestion": (
                        f"Consistent high performance across initial assessments (Mid-Term 1={g1_val}, Mid-Term 2={g2_val}). "
                        "Student is well-positioned for an excellent grade in the End Semester Exam."
                    )
                })
        except (ValueError, TypeError):
            pass

    # 2. Number of Past Failures
    failures = student_data.get("failures")
    if failures is not None:
        try:
            fail_val = int(failures)
            if fail_val >= 2:
                urgent_actions.append({
                    "indicator": f"History of Repeated Class Failures ({fail_val} past failures)",
                    "title": "Structured Academic Recovery Plan",
                    "suggestion": (
                        "Prior course failures represent a significant risk multiplier. Provide access to one-on-one "
                        "tutoring, diagnostic assessment of learning gaps, and customized pace adjustments."
                    ),
                    "action_tag": "Critical Recovery"
                })
            elif fail_val == 1:
                recommended_actions.append({
                    "indicator": "Prior Course Failure (1 failure recorded)",
                    "title": "Targeted Topic Reinforcement",
                    "suggestion": (
                        "A single past failure indicates specific concept vulnerabilities. Ensure active participation "
                        "in weekly office hours and formative diagnostic quizzes."
                    ),
                    "action_tag": "Support Plan"
                })
        except (ValueError, TypeError):
            pass

    # 3. Weekly Study Time
    studytime = student_data.get("studytime")
    if studytime is not None:
        try:
            st_val = int(studytime)
            if st_val == 1:  # < 2 hours per week
                urgent_actions.append({
                    "indicator": "Low Weekly Study Time (< 2 hours/week)",
                    "title": "Establish Structured Daily Study Routine",
                    "suggestion": (
                        "Self-study commitment is under 2 hours weekly. Establish a dedicated study timetable "
                        "with minimum 45-60 minute uninterrupted daily sessions using the Pomodoro technique."
                    ),
                    "action_tag": "Time Management"
                })
            elif st_val == 2:  # 2 to 5 hours
                recommended_actions.append({
                    "indicator": "Moderate Study Time (2 to 5 hours/week)",
                    "title": "Optimize Study Quality & Spaced Repetition",
                    "suggestion": (
                        "Increase study sessions slightly to 5-7 hours weekly during exam periods and incorporate active recall "
                        "and flashcard self-testing rather than passive rereading."
                    ),
                    "action_tag": "Habit Optimization"
                })
            elif st_val >= 3:
                positive_notes.append({
                    "indicator": "Commendable Study Habits (> 5 hours/week)",
                    "title": "Consistent Independent Learning",
                    "suggestion": "Student invests commendable independent study time. Maintain this consistent self-directed momentum."
                })
        except (ValueError, TypeError):
            pass

    # 4. School Absences
    absences = student_data.get("absences")
    if absences is not None:
        try:
            abs_val = int(absences)
            if abs_val > 10:
                urgent_actions.append({
                    "indicator": f"High Rate of School Absences ({abs_val} days)",
                    "title": "Attendance Review & Barrier Consultation",
                    "suggestion": (
                        f"Student has missed {abs_val} classes. High absenteeism strongly correlates with grade decay. "
                        "Conduct an advisory meeting to explore potential transport, health, or personal barriers and compile "
                        "a comprehensive lecture catch-up packet."
                    ),
                    "action_tag": "Attendance Intervention"
                })
            elif abs_val > 5:
                recommended_actions.append({
                    "indicator": f"Moderate Absences ({abs_val} days)",
                    "title": "Catch-up Notes & Attendance Monitoring",
                    "suggestion": "Verify that all missed lecture notes and assignment materials have been retrieved and reviewed."
                })
        except (ValueError, TypeError):
            pass

    # 5. School and Family Support
    schoolsup = str(student_data.get("schoolsup", "")).lower()
    famsup = str(student_data.get("famsup", "")).lower()

    if schoolsup == "no" and ("Low" in predicted_category or "Medium" in predicted_category):
        recommended_actions.append({
            "indicator": "Not Enrolled in Extra School Support",
            "title": "Enroll in Institutional Support Programs",
            "suggestion": (
                "The student is not currently receiving supplemental school-sponsored educational support. "
                "Recommend enrollment in school math/language tutoring labs and peer mentor circles."
            ),
            "action_tag": "School Resources"
        })

    if famsup == "no":
        recommended_actions.append({
            "indicator": "Limited Home Educational Support",
            "title": "Campus-Based Quiet Study & Resource Access",
            "suggestion": (
                "Where home study support is limited, encourage utilizing campus libraries, after-school quiet study spaces, "
                "and online educational discussion forums."
            ),
            "action_tag": "Environment"
        })

    # 6. Alcohol Consumption & Social Balance
    dalc = student_data.get("Dalc")
    walc = student_data.get("Walc")
    try:
        if (dalc is not None and int(dalc) >= 3) or (walc is not None and int(walc) >= 4):
            recommended_actions.append({
                "indicator": "High Weekend/Workday Alcohol Scale",
                "title": "Student Health & Lifestyle Balance",
                "suggestion": (
                    "High alcohol consumption scores may impact cognitive retention and sleep consistency. "
                    "Provide access to campus student wellness and peer counseling resources."
                ),
                "action_tag": "Wellness Support"
            })
    except (ValueError, TypeError):
        pass

    # 7. Higher Education Aspirations
    higher = str(student_data.get("higher", "")).lower()
    if higher == "yes":
        positive_notes.append({
            "indicator": "Aspiration for Higher Education",
            "title": "Strong Academic Motivation",
            "suggestion": (
                "The student expresses a clear ambition to pursue higher education. Connect current course objectives "
                "directly with university prerequisite requirements to foster long-term motivation."
            )
        })

    # Fallback if no specific trigger fired
    if not urgent_actions and not recommended_actions:
        recommended_actions.append({
            "indicator": "General Academic Maintenance",
            "title": "Maintain Balanced Progress",
            "suggestion": (
                "Continue adhering to the current academic schedule, participate actively in class discussions, "
                "and maintain regular communication with course instructors."
            ),
            "action_tag": "Continuous Improvement"
        })

    disclaimer = (
        "Educational Advisory Notice: These recommendations are rule-based heuristics developed from educational "
        "support best practices. They reflect observed behavioral and academic indicators rather than deterministic "
        "causal outputs of the machine learning model. Academic outcomes are influenced by numerous personal, instructional, "
        "and environmental factors."
    )

    return {
        "urgent_actions": urgent_actions,
        "recommended_actions": recommended_actions,
        "positive_notes": positive_notes,
        "total_suggestions": len(urgent_actions) + len(recommended_actions),
        "disclaimer": disclaimer,
    }

