#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

void parallel_pagerank(Graph g, double* solution, double damping, double convergence) {
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  int numNodes = num_nodes(g);
  int numSinkNodes = 0;
  double *solution_new = new double[numNodes];
  int *sink_nodes = new int[numNodes];

  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
    if (outgoing_size(g, i) == 0) {
      sink_nodes[numSinkNodes++] = i;
    }
  }

  bool converged = false;
  while (!converged) {
    double sink_sum = 0;
#pragma omp parallel for reduction(+:sink_sum)
    for (Vertex i = 0; i < numSinkNodes; ++i) {
      sink_sum += solution[sink_nodes[i]];
    }
    sink_sum = damping * sink_sum / numNodes;
#pragma omp parallel for
    for (Vertex v = 0; v < numNodes; ++v) {
      double sum = 0;
      auto in_begin = incoming_begin(g, v);
      auto in_end = incoming_end(g, v);
      for (auto u = in_begin; u < in_end; u++) {
        sum += solution[*u] / outgoing_size(g, *u);
      }
      // score_new[vi] = sum over all nodes vj reachable from incoming edges
      //               { score_old[vj] / number of edges leaving vj  }
      solution_new[v] = sum;
      // score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
      solution_new[v] = (damping * solution_new[v]) + (1.0 - damping) / numNodes;
      // ver all nodes v in graph with no outgoing edges
      //                 { damping *score_new[vi] += sum o score_old[v] / numNodes }
      solution_new[v] += sink_sum;
    }
    // global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
    // converged = (global_diff < convergence)
    double global_diff = 0;
#pragma omp parallel for reduction(+:global_diff)
    for (int i = 0; i < numNodes; ++i) {
      global_diff += fabs(solution_new[i] - solution[i]);
      solution[i] = solution_new[i]; // copy
    }
    converged = global_diff < convergence;
  }
  delete solution_new;
}

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{ 
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }
   */
  parallel_pagerank(g, solution, damping, convergence);
}
