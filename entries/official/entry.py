from entries.template import create_build_f, create_env_f, create_action_space_f
from entries.q_template import create_q_entry


main = create_q_entry(create_build_f, create_env_f, create_action_space_f)



