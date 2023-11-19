#ifndef METRIC_FORMULAS_H
#define METRIC_FORMULAS_H

#define CALC_JAC { \
  E J = float(intersection_size) / float(xadj[u + 1] - xadj[u] + xadj[v + 1] - xadj[v] - intersection_size); \
  emetrics[ptr * 7 + 0] = J; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 0] = J; \
  } \
}

#define CALC_AA { \
  E AA = float(intersection_size_adamic_adar); \
  emetrics[ptr * 7 + 1] = AA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 1] = AA; \
  } \
}

#define CALC_RA { \
  E RA = intersection_size_resource_allocation; \
  emetrics[ptr * 7 + 2] = RA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 2] = RA; \
  } \
}

#define CALC_SI { \
  E SI = float(intersection_size) / float(xadj[u + 1] - xadj[u] + xadj[v + 1] - xadj[v]); \
  emetrics[ptr * 7 + 3] = SI; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 3] = SI; \
  } \
}

#define CALC_CN { \
  E CN = float(intersection_size); \
  emetrics[ptr * 7 + 4] = CN; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 4] = CN; \
  } \
}

#define CALC_SL { \
  E SL = float(intersection_size) / rsqrtf((xadj[u + 1] - xadj[u]) * (xadj[v + 1] - xadj[v])); \
  emetrics[ptr * 7 + 5] = SL; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 5] = SL; \
  } \
}

#define CALC_PA { \
  E PA = (E)(xadj[u + 1] - xadj[u]) * (xadj[v + 1] - xadj[v]); \
  emetrics[ptr * 7 + 6] = PA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 7 + 6] = PA; \
  } \
}

#define SET_INTERSECTION { \
  E INTERSECT = float(intersection_size); \
  emetrics[ptr * 3 + 0] = INTERSECT; /* Index 0 for intersection */ \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 3 + 0] = INTERSECT; \
  } \
}

#define SET_AA { \
  E AA = float(intersection_size_adamic_adar); \
  emetrics[ptr * 3 + 1] = AA; /* Index 1 for AA */ \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 3 + 1] = AA; \
  } \
}

#define SET_RA { \
  E RA = intersection_size_resource_allocation; \
  emetrics[ptr * 3 + 2] = RA; /* Index 2 for RA */ \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr * 3 + 2] = RA; \
  } \
}


#endif