import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
SMALL_MODEL_ID = "gpt2"
BIG_MODEL_ID = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(SMALL_MODEL_ID) 

small_model = GPT2LMHeadModel.from_pretrained(SMALL_MODEL_ID, pad_token_id = tokenizer.eos_token_id).to(device)
big_model = GPT2LMHeadModel.from_pretrained(BIG_MODEL_ID, pad_token_id = tokenizer.eos_token_id).to(device)

def createModelFunction(model):
  """ generates the inference functions, inculding softmax """
  return lambda inputs: adjust_distribution(model(inputs).logits[0])

def adjust_distribution(logits, top_k = 40):
  if top_k is not None:
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float('Inf')
  return logits.softmax(dim=-1)

def sample(p):
  """ gets a distrubtion and returns a sample """
  return torch.multinomial(p, 1)       # slow :(  better: torch.argmax(p)

def autoregressive_sampling(x, model, n):
  """
  Given a t-token prefix x1, ..., xt, predict n next-token distributions sequentially
  """
  t = len(x[0])
  T = len(x[0]) + n

  while t < T:
    
    # generate new sample:
    new_pred = sample(model(x)[-1])

    # concatenate to the sequence
    x = torch.cat((x, new_pred.reshape(1,1)), dim = 1)

    t += 1
  return x


def generate_delta_distribution(p, q):
  """
  p,q: two distribution vectors 
  returns max(0, p-q) after normalization
  """
  delta = p-q 

  # zero out negative values:
  delta[delta < 0] = 0

  # normalized 
  delta /= delta.norm(p=1)
  return delta

@torch.no_grad()
def rejection_sampling(x, small_pred, big_pred, k, n):
  t = len(x[0])
  T = t + n

  # repeat until n new tokens have been generated 
  while t < T:

    # *step 1*: 
    # Given a t-token prefix x1, ..., xt, predict k next-token distributions
    # {ps(xi|xs <i)| for t < i ≤ t + k} sequentially using a small_pred
    xs = x
    for _ in range(k):
      new_pred = sample(small_pred(xs)[-1])
      xs = torch.cat((xs, new_pred.reshape(1,1)), dim = 1)


    # step 2: use the big model to compute {pb} in parallel
    ps = small_pred(xs)     # shape = (t, vocab_size)
    pb = big_pred(xs)       # shape = (t, vocab_size)

    # step 3 - rejection sampling
    counter = 0
    for _ in range(k):
      prev_token = t - 1
      x_ts = xs[0][t]

      # sample r ~ U(0,1), compare to the probs ratio:
      if torch.rand(1).to(device) < (pb[prev_token][x_ts] / ps[prev_token][x_ts]):      # accepting
        t += 1
        counter += 1
        if counter < k:
          x = torch.cat((x, x_ts.reshape(1,1)), dim = 1)
        else:  # all k tokens were accepted 
          new_sample = sample(pb[prev_token])
          x = torch.cat((x, new_sample.reshape(1,1)), dim = 1)

    
      else:      # rejecting
        # sample one additional token
        p_delta = generate_delta_distribution(pb[prev_token], ps[prev_token])
        new_sample = sample(p_delta)
        x = torch.cat((x, new_sample.reshape(1,1)), dim = 1)
        t += 1
        break
    
  return x

small_pred = createModelFunction(small_model)
big_pred = createModelFunction(big_model)

def run_sim_autoregressive(input_str, n):
  """
  run simultion on one prompt. measure the running time of the autoregressive 
  """

  inputs = tokenizer.encode(input_str, return_tensors = "pt").to(device)
  
  # autoregressive runtimes for the small model
  start = time.perf_counter()
  output = autoregressive_sampling(inputs, small_pred, n)
  elapsed_time_small = time.perf_counter() - start
  
  # autoregressive runtimes for the large model
  start = time.perf_counter()
  output = autoregressive_sampling(inputs, big_pred, n)
  elapsed_time_big = time.perf_counter() - start
  return elapsed_time_small, elapsed_time_big

def run_sim_rejection_sampling(input_str, k, n):
  """
  run simultion on one prompt. measure the running time of the rejection_sampling 
  """
  inputs = tokenizer.encode(input_str, return_tensors = "pt").to(device)
  
  #runtimes for the efficient attention algorithm
  start = time.perf_counter()
  output = rejection_sampling(inputs, small_pred, big_pred,k, n)
  elapsed_time = time.perf_counter() - start
  return elapsed_time
