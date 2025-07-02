import json
import time
from datetime import datetime

# Import your existing system components
# Adjust these imports based on your actual file structure
from graph_agent import ChatState, graph

def load_jee_problems(file_path, num_problems=15):
    """Load problems from the JEE JSON dataset file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Your dataset is a list of problem dictionaries
    problems = data[:num_problems] if isinstance(data, list) else [data]
    
    print(f"Sample problem structure:")
    if problems:
        sample = problems[0]
        print(f"  - Index: {sample.get('index')}")
        print(f"  - Subject: {sample.get('subject')}")
        print(f"  - Type: {sample.get('type')}")
        print(f"  - Gold Answer: {sample.get('gold')}")
        print(f"  - Question Length: {len(sample.get('question', ''))}")
    
    return problems

def extract_answer(response_text, correct_answer):
    """Extract the model's answer from response text"""
    if not response_text or not isinstance(response_text, str):
        return "NO_ANSWER_FOUND"
    
    response_lower = response_text.lower()
    response_upper = response_text.upper()
    
    # For single MCQ (A, B, C, D)
    if correct_answer in ['A', 'B', 'C', 'D']:
        # Look for explicit answer patterns
        patterns = [
            f"answer is {correct_answer}",
            f"answer: {correct_answer}",
            f"({correct_answer})",
            f"option {correct_answer}",
            f"choice {correct_answer}"
        ]
        
        for pattern in patterns:
            if pattern.lower() in response_lower:
                return correct_answer
        
        # Look for any option mentioned
        for option in ['A', 'B', 'C', 'D']:
            if f"({option})" in response_text or f"answer is {option.lower()}" in response_lower:
                return option
    
    # For multiple MCQ (like "ABD", "CD")
    elif len(correct_answer) > 1 and all(c in 'ABCD' for c in correct_answer):
        found_options = set()
        
        # Look for individual options
        for option in ['A', 'B', 'C', 'D']:
            patterns = [
                f"({option})",
                f"option {option}",
                f"choice {option}",
                option in response_upper.split()
            ]
            if any(pattern if isinstance(pattern, bool) else pattern in response_text for pattern in patterns):
                found_options.add(option)
        
        if found_options:
            return ''.join(sorted(found_options))
    
    # Last resort: look for the exact answer string
    if correct_answer in response_upper:
        return correct_answer
    
    return "NO_ANSWER_FOUND"

def safe_get_state_value(state, key, default=None):
    """Safely get value from state whether it's a dict or object"""
    if isinstance(state, dict):
        return state.get(key, default)
    else:
        return getattr(state, key, default)

def run_single_problem(problem):
    """Run the LangGraph system on a single physics problem"""
    start_time = time.time()
    
    try:
        # Create initial state with the question
        initial_state = ChatState(input=problem['question'])
        
        # Run the workflow and track execution
        final_state = None
        step_count = 0
        execution_steps = []
        
        for step_result in graph.stream(initial_state):
            step_count += 1
            # Get the current step info
            for node_name, state in step_result.items():
                final_state = state
                execution_steps.append(node_name)
        
        execution_time = time.time() - start_time
        
        if final_state is None:
            return {
                'problem_id': problem['index'],
                'problem_type': problem['type'],
                'subject': problem['subject'],
                'success': False,
                'error': 'No final state returned from workflow',
                'execution_time': execution_time,
                'predicted_answer': 'ERROR',
                'correct_answer': problem['gold'],
                'steps_executed': step_count,
                'execution_path': execution_steps
            }
        
        # Extract output - handle both dict and object formats
        output_text = safe_get_state_value(final_state, 'output', '')
        if not output_text:
            # Try alternative keys that might contain the response
            output_text = (safe_get_state_value(final_state, 'response', '') or 
                          safe_get_state_value(final_state, 'answer', '') or 
                          safe_get_state_value(final_state, 'result', '') or
                          str(final_state))
        
        # Extract predicted answer from the final output
        predicted_answer = extract_answer(output_text, problem['gold'])
        
        # Check if answer is correct
        is_correct = predicted_answer == problem['gold']
        
        return {
            'problem_id': problem['index'],
            'problem_type': problem['type'],
            'subject': problem['subject'],
            'success': True,
            'error': None,
            'execution_time': execution_time,
            'predicted_answer': predicted_answer,
            'correct_answer': problem['gold'],
            'is_correct': is_correct,
            'steps_executed': step_count,
            'execution_path': execution_steps,
            'is_math_detected': safe_get_state_value(final_state, 'ismath', False),
            'llm_math_check': safe_get_state_value(final_state, 'llmcheckmath', False),
            'kb_coverage': safe_get_state_value(final_state, 'is_present_in_kb', False),
            'similarity_score': safe_get_state_value(final_state, 'score', 0.0),
            'response_length': len(str(output_text)),
            'final_state_type': type(final_state).__name__,
            'raw_output': str(output_text)[:500]  # First 500 chars for debugging
        }
        
    except Exception as e:
        return {
            'problem_id': problem['index'],
            'problem_type': problem['type'],
            'subject': problem['subject'],
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time,
            'predicted_answer': 'ERROR',
            'correct_answer': problem['gold'],
            'is_correct': False,
            'steps_executed': step_count,
            'execution_path': execution_steps
        }

def calculate_accuracy(results):
    """Calculate comprehensive accuracy metrics"""
    total = len(results)
    successful_runs = sum(1 for r in results if r['success'])
    correct_answers = sum(1 for r in results if r.get('is_correct', False))
    
    # Break down by problem type
    mcq_single = [r for r in results if r['problem_type'] == 'MCQ']
    mcq_multiple = [r for r in results if r['problem_type'] == 'MCQ(multiple)']
    
    mcq_single_correct = sum(1 for r in mcq_single if r.get('is_correct', False))
    mcq_multiple_correct = sum(1 for r in mcq_multiple if r.get('is_correct', False))
    
    return {
        'total_problems': total,
        'successful_runs': successful_runs,
        'correct_answers': correct_answers,
        'overall_accuracy': correct_answers / total if total > 0 else 0,
        'success_rate': successful_runs / total if total > 0 else 0,
        'error_rate': (total - successful_runs) / total if total > 0 else 0,
        'mcq_single_count': len(mcq_single),
        'mcq_single_correct': mcq_single_correct,
        'mcq_single_accuracy': mcq_single_correct / len(mcq_single) if mcq_single else 0,
        'mcq_multiple_count': len(mcq_multiple),
        'mcq_multiple_correct': mcq_multiple_correct,
        'mcq_multiple_accuracy': mcq_multiple_correct / len(mcq_multiple) if mcq_multiple else 0
    }

def run_benchmark():
    """Main benchmark execution"""
    print("🚀 Starting JEE Benchmark...")
    
    # Configuration
    DATASET_FILE = "J:/Assignment/Aiplanet final/jeebenchmarker/dataset.json"  # Your JSON dataset file
    NUM_PROBLEMS = 5
    
    # Load problems
    print(f"📖 Loading {NUM_PROBLEMS} problems from {DATASET_FILE}...")
    try:
        problems = load_jee_problems(DATASET_FILE, NUM_PROBLEMS)
        print(f"✅ Loaded {len(problems)} problems")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Run benchmark
    results = []
    total_time = 0
    
    for i, problem in enumerate(problems, 1):
        print(f"\n🔄 Processing Problem {i}/{len(problems)}")
        print(f"   📝 ID: {problem['index']} | Type: {problem['type']} | Subject: {problem['subject']}")
        print(f"   🎯 Expected Answer: {problem['gold']}")
        
        result = run_single_problem(problem)
        results.append(result)
        total_time += result['execution_time']
        
        # Show progress
        status = "✅" if result['success'] else "❌"
        answer_check = "✓" if result.get('is_correct', False) else "✗"
        print(f"   {status} Execution: {result['execution_time']:.2f}s | Path: {' → '.join(result.get('execution_path', []))}")
        print(f"   {answer_check} Answer: {result['predicted_answer']}")
        
        if result['error']:
            print(f"   ⚠️  Error: {result['error']}")
        
        if result['success']:
            print(f"   📊 Math detected: {result.get('is_math_detected', False)} | KB used: {result.get('kb_coverage', False)}")
            print(f"   🔍 State type: {result.get('final_state_type', 'Unknown')}")
            
            # Show first few words of output for debugging
            raw_output = result.get('raw_output', '')
            if raw_output:
                preview = raw_output[:100] + "..." if len(raw_output) > 100 else raw_output
                print(f"   📄 Output preview: {preview}")
    
    # Calculate metrics
    metrics = calculate_accuracy(results)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 JEE PHYSICS BENCHMARK RESULTS")
    print("="*70)
    print(f"Total Problems: {metrics['total_problems']}")
    print(f"Successful Executions: {metrics['successful_runs']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Error Rate: {metrics['error_rate']:.2%}")
    print(f"Average Execution Time: {total_time/len(results):.2f}s")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    # Breakdown by problem type
    print(f"\n📋 BREAKDOWN BY PROBLEM TYPE:")
    print(f"Single Choice MCQ: {metrics['mcq_single_correct']}/{metrics['mcq_single_count']} ({metrics['mcq_single_accuracy']:.2%})")
    print(f"Multiple Choice MCQ: {metrics['mcq_multiple_correct']}/{metrics['mcq_multiple_count']} ({metrics['mcq_multiple_accuracy']:.2%})")
    
    # System analysis
    print(f"\n🔍 SYSTEM ANALYSIS:")
    math_detected = sum(1 for r in results if r.get('is_math_detected', False))
    llm_math_check = sum(1 for r in results if r.get('llm_math_check', False))
    kb_coverage = sum(1 for r in results if r.get('kb_coverage', False))
    
    print(f"Math Queries Detected (Embedding): {math_detected}/{len(results)}")
    print(f"Math Queries Detected (LLM): {llm_math_check}/{len(results)}")
    print(f"Knowledge Base Coverage: {kb_coverage}/{len(results)}")
    print(f"Average Similarity Score: {sum(r.get('similarity_score', 0) for r in results)/len(results):.3f}")
    
    # State type analysis
    state_types = {}
    for r in results:
        if r['success']:
            state_type = r.get('final_state_type', 'Unknown')
            state_types[state_type] = state_types.get(state_type, 0) + 1
    
    print(f"\n📊 STATE TYPE ANALYSIS:")
    for state_type, count in state_types.items():
        print(f"   {state_type}: {count}")
    
    # Error analysis
    errors = [r for r in results if not r['success']]
    if errors:
        print(f"\n⚠️  ERROR ANALYSIS:")
        for error in errors:
            print(f"   Problem {error['problem_id']}: {error['error']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"jee_physics_benchmark_{timestamp}.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_file': DATASET_FILE,
        'num_problems': NUM_PROBLEMS,
        'metrics': metrics,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("🎉 Benchmark completed!")

if __name__ == "__main__":
    run_benchmark()