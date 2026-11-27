"""Small, deterministic fictional HR dataset for trying the application."""
import csv
import io


def sample_payloads():
    """Return two joinable CSV tables through the normal upload loading path."""
    employees = io.StringIO()
    performance = io.StringIO()
    people = csv.writer(employees)
    scores = csv.writer(performance)
    people.writerow(['employee_id', 'employee_name', 'department', 'job_level',
                     'years_of_service', 'monthly_salary_sgd'])
    scores.writerow(['employee_id', 'review_year', 'performance_score', 'training_hours'])
    departments = ['Human Resources', 'Finance', 'Operations', 'Technology', 'Customer Service']
    for i in range(1, 31):
        people.writerow([f'E{i:03}', f'Demo Employee {i:02}', departments[(i - 1) % 5],
                         ['Associate', 'Senior', 'Manager'][(i - 1) % 3],
                         1 + (i * 3) % 12, 3000 + (i % 10) * 450])
        scores.writerow([f'E{i:03}', 2025, round(2.5 + (i % 6) * 0.5, 1), 8 + (i % 8) * 4])
    return [('demo_employees.csv', employees.getvalue().encode()),
            ('demo_performance.csv', performance.getvalue().encode())]
