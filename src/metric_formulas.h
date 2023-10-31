#ifndef METRIC_FORMULAS_H
#define METRIC_FORMULAS_H

#define CALC_JAC { \
  E J = float(intersection_size) / float(xadj[u + 1] - xadj[u] + xadj[v + 1] - xadj[v] - intersection_size); \
  emetrics[ptr] = J; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr] = J; \
  } \
}

#define CALC_AA { \
  E AA = float(intersection_size_adamic_adar); \
  emetrics[ptr + 1] = AA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 1] = AA; \
  } \
}

#define CALC_RA { \
  E RA = intersection_size_resource_allocation; \
  emetrics[ptr + 6] = RA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 6] = RA; \
  } \
}

#define CALC_SI { \
  E SI = float(intersection_size) / float(xadj[u + 1] - xadj[u] + xadj[v + 1] - xadj[v]); \
  emetrics[ptr + 2 ] = SI; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 2] = SI; \
  } \
}

#define CALC_CN { \
  E CN = float(intersection_size); \
  emetrics[ptr + 3] = CN; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 3] = CN; \
  } \
}

#define CALC_SL { \
  E SL = float(intersection_size) / rsqrtf((xadj[u + 1] - xadj[u]) * (xadj[v + 1] - xadj[v])); \
  emetrics[ptr + 4] = SL; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 4] = SL; \
  } \
}

#define CALC_PA { \
  E PA = (E)(xadj[u + 1] - xadj[u]) * (xadj[v + 1] - xadj[v]); \
  emetrics[ptr + 5] = PA; \
  if (other_ptr != (EN)-1) { \
    emetrics[other_ptr + 5] = PA; \
  } \
}

#define SET_INTERSECTION { \
    E INTERSECT = float(intersection_size); \
    emetrics[ptr + 3] = INTERSECT; \
    if (other_ptr != (EN)-1) { \
        emetrics[other_ptr] = INTERSECT; \
    } \
}

#define SET_AA { \
    E AA = float(intersection_size_adamic_adar); \
    emetrics[ptr + 1] = AA; \
    if (other_ptr != (EN)-1) { \
        emetrics[other_ptr + 1] = AA; \
    } \
}

#define SET_RA { \
    E RA = intersection_size_resource_allocation; \
    emetrics[ptr + 6] = RA; \
    if (other_ptr != (EN)-1) { \
        emetrics[other_ptr + 2] = RA; \
    } \
}


#endif