def before_all(context):
    # Set up any global test configuration
    pass

def after_all(context):
    # Clean up after all tests
    pass

def before_feature(context, feature):
    # Set up feature-specific test configuration
    pass

def after_feature(context, feature):
    # Clean up after each feature
    pass

def before_scenario(context, scenario):
    # Reset tokenizer and other objects before each scenario
    context.tokenizer = None
    context.tokens = None
    context.token_ids = None
    context.decoded_text = None
    context.vocab = None
    context.corpus = None
    context.special_tokens = None

def after_scenario(context, scenario):
    # Clean up after each scenario
    pass